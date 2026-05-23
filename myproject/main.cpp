#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>

namespace fs = std::filesystem;

// Funzione intatta per estrarre le feature dal contorno
void extractAndSaveFeatures(const cv::Mat& imgOriginale, const cv::Mat& mask,
    const std::string& cellType, const std::string& imageName,
    std::ofstream& csvFile, cv::Mat& imgAnteprima, double minArea = 5.0) {

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < minArea) continue;

        double perimetro = cv::arcLength(contour, true);
        cv::Rect boundingBox = cv::boundingRect(contour);
        double aspectRatio = (double)boundingBox.width / (double)boundingBox.height;

        double circolarita = 0.0;
        if (perimetro > 0) {
            circolarita = (4.0 * CV_PI * area) / (perimetro * perimetro);
        }

        cv::Mat singleCellMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(singleCellMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
        cv::Scalar meanColor = cv::mean(imgOriginale, singleCellMask);

        // Colori per il riquadro a schermo
        cv::Scalar colorBox(0, 255, 0);
        if (cellType == "GlobuloBianco") colorBox = cv::Scalar(255, 0, 0);
        else if (cellType == "GlobuloRosso") colorBox = cv::Scalar(0, 0, 255);
        else if (cellType == "Piastrina") colorBox = cv::Scalar(0, 255, 255);

        // Disegno in tempo reale e etichetta
        cv::rectangle(imgAnteprima, boundingBox, colorBox, 2);
        cv::putText(imgAnteprima, cellType.substr(0, 3), cv::Point(boundingBox.x, std::max(0, boundingBox.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, colorBox, 1);

        csvFile << imageName << ","
            << cellType << ","
            << boundingBox.x << ","
            << boundingBox.y << ","
            << boundingBox.width << ","
            << boundingBox.height << ","
            << area << ","
            << perimetro << ","
            << circolarita << ","
            << aspectRatio << ","
            << meanColor[0] << ","
            << meanColor[1] << ","
            << meanColor[2] << "\n";
    }
}

int main() {
    try {
        std::string folderOriginali = "example_images/";
        std::string folderAnnotate = "output/";
        std::string outFolderBianchi = "output_bianchi/";
        std::string outFolderPiastrine = "output_piastrine/";
        std::string outFolderRossi = "output_rossi/";

        // ---------------------------------------------------------
        // 1. PERCORSO CSV ASSOLUTO (Sovrascrive sempre il file esistente)
        // ---------------------------------------------------------
        std::string csvPath = "C:\\Progetti\\Template C++\\ProgettoML\\features_cellule.csv";

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

        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // Apertura in modalità standard (tronca/sovrascrive il file precedente)
        std::ofstream csvFile(csvPath);
        if (!csvFile.is_open()) {
            std::cerr << "ERRORE: Impossibile creare il file CSV in: " << csvPath << std::endl;
            return -1;
        }
        csvFile << "ImageName,CellType,BoxX,BoxY,BoxW,BoxH,Area,Perimeter,Circularity,AspectRatio,MeanBlue,MeanGreen,MeanRed\n";

        for (size_t f = 0; f < imagePaths.size(); f++) {
            cv::Mat imgOriginale = cv::imread(imagePaths[f], cv::IMREAD_COLOR);
            if (imgOriginale.empty()) continue;

            cv::Mat imgAnteprima = imgOriginale.clone();

            std::string fullPath = imagePaths[f];
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string fileName = fullPath.substr(lastSlash + 1);

            cv::Mat imgAnnotataReale = cv::imread(folderAnnotate + fileName, cv::IMREAD_COLOR);
            if (imgAnnotataReale.empty()) imgAnnotataReale = imgOriginale.clone();

            cv::Mat imgMedian, imgBilateral;
            cv::medianBlur(imgOriginale, imgMedian, 3);
            cv::bilateralFilter(imgMedian, imgBilateral, 9, 75, 75);

            cv::Mat imgHSV, imgGray;
            cv::cvtColor(imgBilateral, imgHSV, cv::COLOR_BGR2HSV);
            cv::cvtColor(imgBilateral, imgGray, cv::COLOR_BGR2GRAY);

            // ==========================================
            // SEZIONE GLOBULI BIANCHI
            // ==========================================
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

            // ==========================================
            // SEZIONE PIASTRINE
            // ==========================================
            cv::Mat imgGreen;
            cv::extractChannel(imgOriginale, imgGreen, 1);
            cv::Mat blackHat;
            cv::Mat kernelBH = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
            cv::morphologyEx(imgGreen, blackHat, cv::MORPH_BLACKHAT, kernelBH);
            cv::Mat maskPiastrineRaw;
            cv::threshold(blackHat, maskPiastrineRaw, 25, 255, cv::THRESH_BINARY);

            cv::Mat areaDaEscludere;
            cv::dilate(maskSoloBianchi, areaDaEscludere, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));
            cv::Mat maskSfondoLibero;
            cv::bitwise_not(areaDaEscludere, maskSfondoLibero);
            cv::bitwise_and(maskPiastrineRaw, maskSfondoLibero, maskPiastrineRaw);

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

            // ==========================================
            // SEZIONE GLOBULI ROSSI
            // ==========================================
            cv::Mat maskTutteLeCellule;
            cv::threshold(imgGray, maskTutteLeCellule, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

            cv::Mat kernelBianchiGrande = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
            cv::Mat maskBianchiDilatata;
            cv::dilate(maskSoloBianchi, maskBianchiDilatata, kernelBianchiGrande);

            cv::Mat maskRosa;
            cv::subtract(maskTutteLeCellule, maskBianchiDilatata, maskRosa);

            std::vector<std::vector<cv::Point>> contoursRosa;
            cv::findContours(maskRosa, contoursRosa, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Mat maskRosaTmp = cv::Mat::zeros(maskRosa.size(), CV_8UC1);
            cv::drawContours(maskRosaTmp, contoursRosa, -1, cv::Scalar(255), cv::FILLED);
            cv::morphologyEx(maskRosaTmp, maskRosaTmp, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

            cv::Mat kernelErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::Mat maskEroded;
            cv::erode(maskRosaTmp, maskEroded, kernelErode, cv::Point(-1, -1), 2);

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

            int step = 40;
            std::vector<cv::Point> listaCentri;
            int latoRettangolo = 80;
            int distanzaMinima = 50;

            for (size_t i = 0; i < skelPoints.size(); i += step) {
                cv::Point pt = skelPoints[i];
                if (maskRosa.at<uchar>(pt.y, pt.x) == 0) continue;

                bool troppoVicino = false;
                for (const auto& centroGiaSalvato : listaCentri) {
                    if (cv::norm(pt - centroGiaSalvato) < distanzaMinima) {
                        troppoVicino = true;
                        break;
                    }
                }

                if (!troppoVicino) {
                    listaCentri.push_back(pt);
                }
            }

            // SALVATAGGIO DELLE MASCHERE
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);
            cv::imwrite(outFolderRossi + fileName, maskRosa);

            // =====================================================================
            // ESTRAZIONE FEATURE FINALE
            // =====================================================================
            extractAndSaveFeatures(imgOriginale, maskSoloBianchi, "GlobuloBianco", fileName, csvFile, imgAnteprima);
            extractAndSaveFeatures(imgOriginale, maskSoloPiastrine, "Piastrina", fileName, csvFile, imgAnteprima);

            // ---------------------------------------------------------
            // 2. SOLUZIONE GLOBULI ROSSI: RETTANGOLI STRETTI E ADERENTI
            // ---------------------------------------------------------
            for (size_t i = 0; i < listaCentri.size(); i++) {
                cv::Mat maskSingoloRettangolo = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                cv::Point pt = listaCentri[i];

                int x = std::max(0, pt.x - latoRettangolo / 2);
                int y = std::max(0, pt.y - latoRettangolo / 2);
                int w = std::min(imgOriginale.cols - x, latoRettangolo);
                int h = std::min(imgOriginale.rows - y, latoRettangolo);

                // Disegniamo lo stampino finto (che serve solo come limite di ricerca)
                cv::rectangle(maskSingoloRettangolo, cv::Rect(x, y, w, h), cv::Scalar(255), cv::FILLED);

                cv::Mat porzioneRossoReale;
                cv::bitwise_and(maskRosa, maskSingoloRettangolo, porzioneRossoReale);

                // Troviamo i contorni ESATTI della cellula dentro quello stampino
                std::vector<std::vector<cv::Point>> localContours;
                cv::findContours(porzioneRossoReale, localContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                for (const auto& localContour : localContours) {
                    double localArea = cv::contourArea(localContour);

                    // Solo se il pezzo trovato è grande abbastanza per essere un globulo rosso...
                    if (localArea >= 300.0) {
                        cv::Mat maskStrettaDefinitiva = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                        cv::drawContours(maskStrettaDefinitiva, std::vector<std::vector<cv::Point>>{localContour}, -1, cv::Scalar(255), cv::FILLED);

                        // ...lo mandiamo alla funzione, che calcolerà il boundingRect REALE su questo contorno ristretto!
                        extractAndSaveFeatures(imgOriginale, maskStrettaDefinitiva, "GlobuloRosso", fileName, csvFile, imgAnteprima, 300.0);
                    }
                }
            }
            // -----------------------------------------------------------------

            // Dashboard
            cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
            cv::imshow("1. GUIDA REALE", imgAnnotataReale);

            cv::namedWindow("7. RISULTATI DA INVIARE AL CSV", cv::WINDOW_NORMAL);
            cv::imshow("7. RISULTATI DA INVIARE AL CSV", imgAnteprima);

            int key = cv::waitKey(0);
            if (key == 27) break;
        }
        std::cout << "\n[FINE] Pipeline completata! Dataset salvato sovrascrivendo: " << csvPath << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}