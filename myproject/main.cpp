#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>

namespace fs = std::filesystem;

// AGGIORNATO: Aggiunto il parametro minArea con valore di default a 5.0
void extractAndSaveFeatures(const cv::Mat& imgOriginale, const cv::Mat& mask,
    const std::string& cellType, const std::string& imageName,
    std::ofstream& csvFile, double minArea = 5.0) {

    std::vector<std::vector<cv::Point>> contours;
    // Troviamo i contorni esterni nella maschera
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);

        // Filtriamo in base all'area minima richiesta (5.0 per bianchi/piastrine, 300.0 per i rossi)
        if (area < minArea) continue;

        double perimetro = cv::arcLength(contour, true);

        // Bounding box per calcolare l'Aspect Ratio
        cv::Rect boundingBox = cv::boundingRect(contour);
        double aspectRatio = (double)boundingBox.width / (double)boundingBox.height;

        // Circolarità: 4 * pi * Area / (Perimetro^2).
        double circolarita = 0.0;
        if (perimetro > 0) {
            circolarita = (4.0 * CV_PI * area) / (perimetro * perimetro);
        }

        // Estrazione del colore medio: creiamo una maschera solo per questa singola cellula
        cv::Mat singleCellMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(singleCellMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        cv::Scalar meanColor = cv::mean(imgOriginale, singleCellMask);

        // Salviamo i dati nel CSV
        csvFile << imageName << ","
            << cellType << ","
            << area << ","
            << perimetro << ","
            << circolarita << ","
            << aspectRatio << ","
            << meanColor[0] << "," // Blu medio
            << meanColor[1] << "," // Verde medio
            << meanColor[2] << "\n"; // Rosso medio
    }
}

int main() {
    try {
        // =========================================================================
        // 0. SETUP PERCORSI
        // =========================================================================
        std::string folderOriginali = "example_images/";
        std::string folderAnnotate = "output/";
        std::string outFolderBianchi = "output_bianchi/";
        std::string outFolderPiastrine = "output_piastrine/";
        std::string outFolderRossi = "output_rossi/";

        std::string csvPath = "features_cellule.csv";

        fs::create_directories(outFolderBianchi);
        fs::create_directories(outFolderPiastrine);
        fs::create_directories(outFolderRossi);

        std::vector<cv::String> imagePaths;
        cv::glob(folderOriginali + "*.jpeg", imagePaths);
        if (imagePaths.empty()) cv::glob(folderOriginali + "*.jpg", imagePaths);

        if (imagePaths.empty()) {
            std::cerr << "ERRORE: Nessuna immagine trovata in example_images." << std::endl;
            return -1;
        }

        // Parametri per i globuli bianchi
        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // SETUP FILE CSV PER LE FEATURE
        std::ofstream csvFile(csvPath);
        if (!csvFile.is_open()) {
            std::cerr << "ERRORE: Impossibile creare il file CSV." << std::endl;
            return -1;
        }
        csvFile << "ImageName,CellType,Area,Perimeter,Circularity,AspectRatio,MeanBlue,MeanGreen,MeanRed\n";

        // =========================================================================
        // CICLO DI ELABORAZIONE IMMAGINI
        // =========================================================================
        for (size_t f = 0; f < imagePaths.size(); f++) {
            cv::Mat imgOriginale = cv::imread(imagePaths[f], cv::IMREAD_COLOR);
            if (imgOriginale.empty()) continue;

            std::string fullPath = imagePaths[f];
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string fileName = fullPath.substr(lastSlash + 1);

            cv::Mat imgAnnotataReale = cv::imread(folderAnnotate + fileName, cv::IMREAD_COLOR);
            if (imgAnnotataReale.empty()) imgAnnotataReale = imgOriginale.clone();

            // --- PRE-PROCESSING ---
            cv::Mat imgMedian, imgBilateral;
            cv::medianBlur(imgOriginale, imgMedian, 3);
            cv::bilateralFilter(imgMedian, imgBilateral, 9, 75, 75);

            cv::Mat imgHSV, imgGray;
            cv::cvtColor(imgBilateral, imgHSV, cv::COLOR_BGR2HSV);
            cv::cvtColor(imgBilateral, imgGray, cv::COLOR_BGR2GRAY);

            // =====================================================================
            // 1. SEZIONE: GLOBULI BIANCHI
            // =====================================================================
            cv::Mat maskViolaGlobale;
            cv::inRange(imgHSV, lowerViolaGlobale, upperViolaGlobale, maskViolaGlobale);

            cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
            cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

            cv::Mat maskSoloBianchi = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
            cv::Mat labelsB, statsB, centroidsB;
            int nLabelsB = cv::connectedComponentsWithStats(maskViolaGlobale, labelsB, statsB, centroidsB);

            for (int i = 1; i < nLabelsB; i++) {
                if (statsB.at<int>(i, cv::CC_STAT_AREA) >= 800) {
                    maskSoloBianchi.setTo(255, labelsB == i);
                }
            }
            cv::morphologyEx(maskSoloBianchi, maskSoloBianchi, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

            // =====================================================================
            // 2. SEZIONE: PIASTRINE
            // =====================================================================
            cv::Mat imgGreen;
            cv::extractChannel(imgOriginale, imgGreen, 1);

            cv::Mat blackHat;
            cv::Mat kernelBH = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
            cv::morphologyEx(imgGreen, blackHat, cv::MORPH_BLACKHAT, kernelBH);

            cv::Mat maskPiastrineRaw;
            cv::threshold(blackHat, maskPiastrineRaw, 25, 255, cv::THRESH_BINARY);

            cv::Mat maskSoloPiastrine = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
            cv::Mat labelsP, statsP, centroidsP;
            int nLabelsP = cv::connectedComponentsWithStats(maskPiastrineRaw, labelsP, statsP, centroidsP);

            for (int i = 1; i < nLabelsP; i++) {
                int area = statsP.at<int>(i, cv::CC_STAT_AREA);
                if (area >= 6 && area <= 300) {
                    maskSoloPiastrine.setTo(255, labelsP == i);
                }
            }

            cv::Mat maskPiastrineVis;
            cv::dilate(maskSoloPiastrine, maskPiastrineVis, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

            // =====================================================================
             // 3. SEZIONE: GLOBULI ROSSI (CORRETTA L'ORDINE DELLE SOTTRAZIONI)
             // =====================================================================
            cv::Mat maskTutteLeCellule;
            cv::threshold(imgGray, maskTutteLeCellule, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

            // 1. SOTTRAIAMO I BIANCHI IMMEDIATAMENTE (Meno aggressivo: 31x31 invece di 40x40)
            cv::Mat kernelBianchiGrande = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
            cv::Mat maskBianchiDilatata;
            cv::dilate(maskSoloBianchi, maskBianchiDilatata, kernelBianchiGrande);

            // Otteniamo subito la maschera reale depurata dai bianchi
            cv::Mat maskRosa;
            cv::subtract(maskTutteLeCellule, maskBianchiDilatata, maskRosa);

            // 2. RIEMPIMENTO DEI BUCHI SULLA MASCHERA GIA' PULITA
            std::vector<std::vector<cv::Point>> contoursRosa;
            cv::findContours(maskRosa, contoursRosa, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Mat maskRosaTmp = cv::Mat::zeros(maskRosa.size(), CV_8UC1);
            cv::drawContours(maskRosaTmp, contoursRosa, -1, cv::Scalar(255), cv::FILLED);
            cv::morphologyEx(maskRosaTmp, maskRosaTmp, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

            // 3. EROSIONE PRE-SKELETON (Ridotta a 2 iterazioni per non cancellare cellule piccole)
            cv::Mat kernelErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::Mat maskEroded;
            cv::erode(maskRosaTmp, maskEroded, kernelErode, cv::Point(-1, -1), 2);

            // 4. CALCOLO DELLO SKELETON (Ora non calcolerà mai lo skeleton sui bianchi!)
            cv::Mat skel = cv::Mat::zeros(maskEroded.size(), CV_8UC1);
            cv::Mat temp, eroded;
            cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
            cv::Mat imgSkel = maskEroded.clone();
            bool done = false;

            while (!done) {
                cv::erode(imgSkel, eroded, element);
                cv::dilate(eroded, temp, element);
                cv::subtract(imgSkel, temp, temp);
                cv::bitwise_or(skel, temp, skel);
                imgSkel = eroded.clone();
                if (cv::countNonZero(imgSkel) == 0) done = true;
            }

            std::vector<cv::Point> skelPoints;
            cv::findNonZero(skel, skelPoints);

            // 5. CAMPIONAMENTO DEI CENTRI E CREAZIONE RETTANGOLI
            int step = 40; // Puoi abbassare di nuovo lo step, tanto ora filtra per distanza!
            std::vector<cv::Point> listaCentri;
            cv::Mat maskRossiStimati = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);

            int latoRettangolo = 80;
            int distanzaMinima = 50; // <-- NOVITÀ: Distanza minima in pixel tra due rettangoli

            for (size_t i = 0; i < skelPoints.size(); i += step) {
                cv::Point pt = skelPoints[i];

                // 1. Controllo sfondo nero (Niente rettangoli nel vuoto)
                if (maskRosa.at<uchar>(pt.y, pt.x) == 0) continue;

                // 2. Controllo Distanza (Niente cloni sulla stessa cellula)
                bool troppoVicino = false;
                for (const auto& centroGiaSalvato : listaCentri) {
                    // cv::norm calcola la distanza in linea d'aria tra due punti
                    if (cv::norm(pt - centroGiaSalvato) < distanzaMinima) {
                        troppoVicino = true;
                        break;
                    }
                }

                // Salviamo il punto solo se ha superato entrambi i test
                if (!troppoVicino) {
                    listaCentri.push_back(pt);

                    // Creazione sicura del rettangolo
                    int x = std::max(0, pt.x - latoRettangolo / 2);
                    int y = std::max(0, pt.y - latoRettangolo / 2);
                    int w = std::min(imgOriginale.cols - x, latoRettangolo);
                    int h = std::min(imgOriginale.rows - y, latoRettangolo);

                    cv::rectangle(maskRossiStimati, cv::Rect(x, y, w, h), cv::Scalar(255), cv::FILLED);
                }
            }

            // =====================================================================
            // SALVATAGGIO DELLE MASCHERE GLOBALI FINALI
            // =====================================================================
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);
            cv::imwrite(outFolderRossi + fileName, maskRosa);

            // =====================================================================
            // ESTRAZIONE FEATURE
            // =====================================================================
            // 1. Estrazione classica per Bianchi e Piastrine
            extractAndSaveFeatures(imgOriginale, maskSoloBianchi, "GlobuloBianco", fileName, csvFile);
            extractAndSaveFeatures(imgOriginale, maskSoloPiastrine, "Piastrina", fileName, csvFile);

            // 2. NUOVO APPROCCIO: Estrazione Globuli Rossi con stampino RETTANGOLARE locale
            for (size_t i = 0; i < listaCentri.size(); i++) {
                cv::Mat maskSingoloRettangolo = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                cv::Point pt = listaCentri[i];

                int x = std::max(0, pt.x - latoRettangolo / 2);
                int y = std::max(0, pt.y - latoRettangolo / 2);
                int w = std::min(imgOriginale.cols - x, latoRettangolo);
                int h = std::min(imgOriginale.rows - y, latoRettangolo);

                cv::rectangle(maskSingoloRettangolo, cv::Rect(x, y, w, h), cv::Scalar(255), cv::FILLED);

                cv::Mat porzioneRossoReale;
                cv::bitwise_and(maskRosa, maskSingoloRettangolo, porzioneRossoReale);

                extractAndSaveFeatures(imgOriginale, porzioneRossoReale, "GlobuloRosso", fileName, csvFile, 300.0);
            }

            // =====================================================================
            // DASHBOARD DI VISUALIZZAZIONE
            // =====================================================================
            cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
            cv::imshow("1. GUIDA REALE", imgAnnotataReale);

            cv::namedWindow("2. MASK BIANCHI", cv::WINDOW_NORMAL);
            cv::imshow("2. MASK BIANCHI", maskSoloBianchi);

            cv::namedWindow("3. MASK PIASTRINE", cv::WINDOW_NORMAL);
            cv::imshow("3. MASK PIASTRINE", maskPiastrineVis);

            // Questa finestra non serve più, commentata per pulizia visiva
            // cv::namedWindow("4. RETTANGOLI IDEALI ROSSI", cv::WINDOW_NORMAL);
            // cv::imshow("4. RETTANGOLI IDEALI ROSSI", maskRossiStimati);

            cv::namedWindow("5. MASK ROSSI FINALE (FORME REALI)", cv::WINDOW_NORMAL);
            cv::imshow("5. MASK ROSSI FINALE (FORME REALI)", maskRosa);

            cv::Mat imgVisualizzazioneStampino = cv::Mat::zeros(imgOriginale.size(), CV_8UC3);
            cv::cvtColor(maskRosa, imgVisualizzazioneStampino, cv::COLOR_GRAY2BGR);
            for (const auto& pt : listaCentri) {
                int x = std::max(0, pt.x - latoRettangolo / 2);
                int y = std::max(0, pt.y - latoRettangolo / 2);
                int w = std::min(imgOriginale.cols - x, latoRettangolo);
                int h = std::min(imgOriginale.rows - y, latoRettangolo);

                cv::rectangle(imgVisualizzazioneStampino, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
            }
            cv::namedWindow("6. ANTEPRIMA STAMPINI SUL REALE", cv::WINDOW_NORMAL);
            cv::imshow("6. ANTEPRIMA STAMPINI SUL REALE", imgVisualizzazioneStampino);

            int key = cv::waitKey(0);
            if (key == 27) break; // ESC per uscire
        }
        std::cout << "\n[FINE] Pipeline completata! Dataset salvato in " << csvPath << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}