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

        cv::Scalar colorBox(0, 255, 0);
        if (cellType == "GlobuloBianco") colorBox = cv::Scalar(255, 0, 0);
        else if (cellType == "GlobuloRosso") colorBox = cv::Scalar(0, 0, 255);
        else if (cellType == "Piastrina") colorBox = cv::Scalar(0, 255, 255);

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

// Struttura per gestire multipli dataset in modo pulito
struct DatasetConfig {
    std::string inputFolder;
    std::string outputCsv;
    std::string nomeFase;
};

int main() {
    try {
        // =========================================================================
        // 1. CONFIGURAZIONE DEI PERCORSI ASSOLUTI (CORRETTI)
        // =========================================================================
        // Nota lo slash finale '/' per evitare che i file si accavallino al nome della cartella
        //std::string cartellaProgettoML = "C:/Template-C-/ProgettoML/csv/";
        std::string cartellaProgettoML = "../../ProgettoML/csv/";

        // Creiamo la directory (create_directories ignora lo slash finale, quindi funziona perfettamente)
        fs::create_directories(cartellaProgettoML);

        std::vector<DatasetConfig> pipeline = {
            // FASE 1: Dati di addestramento (anche qui, slash finale fondamentale per il cv::glob dopo!)
            //{"C:/Template-C-/ProgettoIPA/archive/train/img/", cartellaProgettoML + "features_cellule_train.csv", "TRAIN"},

           {"../archive/train/img/", cartellaProgettoML + "features_cellule_train.csv", "TRAIN"},


           // FASE 2: Dati di test 
           //{"C:/Template-C-/ProgettoIPA/archive/test/img/", cartellaProgettoML + "features_cellule_test.csv", "TEST"}
           {"../archive/test/img/", cartellaProgettoML + "features_cellule_test.csv", "TEST"}
        };

        // Occhio a questa se non trova le immagini annotate, nel caso metti il percorso assoluto anche qui!
        std::string folderAnnotate = "../output/";

        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // =========================================================================
        // 2. CICLO SUI DATASET (Prima elabora il Train, poi il Test)
        // =========================================================================
        for (const auto& dataset : pipeline) {
            std::cout << "\n=======================================================" << std::endl;
            std::cout << " INIZIO ELABORAZIONE FASE: " << dataset.nomeFase << std::endl;
            std::cout << " Lettura da: " << dataset.inputFolder << std::endl;
            std::cout << " Scrittura su: " << dataset.outputCsv << std::endl;
            std::cout << "=======================================================\n" << std::endl;

            std::vector<cv::String> imagePaths;
            cv::glob(dataset.inputFolder + "*.jpeg", imagePaths);
            if (imagePaths.empty()) cv::glob(dataset.inputFolder + "*.jpg", imagePaths);

            if (imagePaths.empty()) {
                std::cerr << "[WARNING] Nessuna immagine trovata in " << dataset.inputFolder << ". Passo alla fase successiva." << std::endl;
                continue;
            }

            // APERTURA FILE CSV CON FLAG TRUNC
            std::ofstream csvFile(dataset.outputCsv, std::ios::out | std::ios::trunc);
            if (!csvFile.is_open()) {
                std::cerr << "[ERRORE] Impossibile creare il file " << dataset.outputCsv << std::endl;
                continue;
            }
            csvFile << "ImageName,CellType,BoxX,BoxY,BoxW,BoxH,Area,Perimeter,Circularity,AspectRatio,MeanBlue,MeanGreen,MeanRed\n";

            // --- INIZIO AGGIUNTA 1: CREAZIONE DEL FILE CSV PER IL CLASSIFICATORE BINARIO ---
            std::string outputCsvBinario = dataset.outputCsv;
            size_t posCsv = outputCsvBinario.find(".csv");
            if (posCsv != std::string::npos) outputCsvBinario.insert(posCsv, "_BINARIO");

            std::ofstream csvBinario(outputCsvBinario, std::ios::out | std::ios::trunc);
            if (csvBinario.is_open()) {
                csvBinario << "ImageName,CellType,BoxX,BoxY,BoxW,BoxH,Area,Perimeter,Circularity,AspectRatio,MeanBlue,MeanGreen,MeanRed\n";
            }
            // --- FINE AGGIUNTA 1 ---

            // VARIABILE DI CONTROLLO: True = Mostra immagini. False = Lavora solo in background
            bool mostraFinestre = true;

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

                // --- INIZIO AGGIUNTA 2: ESTRAZIONE MASCHERE FOREGROUND E BACKGROUND ---
                cv::Mat maskAdattiva, maskForeground, maskBackground;

                // Usiamo il threshold adattivo: calcola la soglia localmente! (molto meglio per i vetrini)
                cv::adaptiveThreshold(imgGray, maskAdattiva, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 101, 5);

                maskForeground = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                maskBackground = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);

                std::vector<std::vector<cv::Point>> contoursBin;
                cv::findContours(maskAdattiva, contoursBin, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                for (size_t i = 0; i < contoursBin.size(); i++) {
                    double areaBin = cv::contourArea(contoursBin[i]);

                    // Selezioniamo le soglie reali per questa immagine
                    if (areaBin >= 150.0) {
                        // Dimensioni da cellula o grumo valido -> Foreground
                        cv::drawContours(maskForeground, contoursBin, (int)i, cv::Scalar(255), cv::FILLED);
                    }
                    else if (areaBin >= 10.0 && areaBin < 150.0) {
                        // Dimensioni troppo piccole per essere cellule -> Rumore (Background per il ML)
                        cv::drawContours(maskBackground, contoursBin, (int)i, cv::Scalar(255), cv::FILLED);
                    }
                }

                if (csvBinario.is_open()) {
                    cv::Mat anteprimaFantasma = imgOriginale.clone(); // Immagine dummy per assorbire i quadrati verdi
                    extractAndSaveFeatures(imgOriginale, maskForeground, "Foreground", fileName, csvBinario, anteprimaFantasma, 150.0);
                    extractAndSaveFeatures(imgOriginale, maskBackground, "Background", fileName, csvBinario, anteprimaFantasma, 10.0);
                }
                // --- FINE AGGIUNTA 2 ---

                // --- BIANCHI ---
                cv::Mat maskViolaGlobale;
                cv::inRange(imgHSV, lowerViolaGlobale, upperViolaGlobale, maskViolaGlobale);
                cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
                cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

                cv::Mat maskSoloBianchi = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                cv::Mat labelsB, statsB, centroidsB;
                int nLabelsB = cv::connectedComponentsWithStats(maskViolaGlobale, labelsB, statsB, centroidsB);
                for (int i = 1; i < nLabelsB; i++) {

                    if (statsB.at<int>(i, cv::CC_STAT_AREA) >= 1500) {
                        maskSoloBianchi.setTo(255, labelsB == i);
                    }
                }
                cv::morphologyEx(maskSoloBianchi, maskSoloBianchi, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

                // --- PIASTRINE ---
                cv::Mat imgGreen;
                cv::extractChannel(imgOriginale, imgGreen, 1);
                cv::Mat blackHat;
                cv::Mat kernelBH = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(21, 21));
                cv::morphologyEx(imgGreen, blackHat, cv::MORPH_BLACKHAT, kernelBH);
                cv::Mat maskPiastrineRaw;
                cv::threshold(blackHat, maskPiastrineRaw, 30, 255, cv::THRESH_BINARY);

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
                    if (area >= 15 && area <= 300) {
                        maskSoloPiastrine.setTo(255, labelsP == i);
                    }
                }
                cv::Mat maskPiastrineVis;
                cv::dilate(maskSoloPiastrine, maskPiastrineVis, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

                // --- ROSSI ---
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

                // --- ESTRAZIONE FINALE ---
                extractAndSaveFeatures(imgOriginale, maskSoloBianchi, "GlobuloBianco", fileName, csvFile, imgAnteprima, 1500.0);
                extractAndSaveFeatures(imgOriginale, maskSoloPiastrine, "Piastrina", fileName, csvFile, imgAnteprima);

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

                    std::vector<std::vector<cv::Point>> localContours;
                    cv::findContours(porzioneRossoReale, localContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    for (const auto& localContour : localContours) {
                        double localArea = cv::contourArea(localContour);
                        if (localArea >= 300.0) {
                            cv::Mat maskStrettaDefinitiva = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                            cv::drawContours(maskStrettaDefinitiva, std::vector<std::vector<cv::Point>>{localContour}, -1, cv::Scalar(255), cv::FILLED);
                            extractAndSaveFeatures(imgOriginale, maskStrettaDefinitiva, "GlobuloRosso", fileName, csvFile, imgAnteprima, 300.0);
                        }
                    }
                }

                // --- GESTIONE VISIVA INTELLIGENTE ---
                if (mostraFinestre) {
                    cv::putText(imgAnteprima, dataset.nomeFase, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
                    cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
                    cv::imshow("1. GUIDA REALE", imgAnnotataReale);

                    // --- INIZIO AGGIUNTA 3: VISUALIZZAZIONE MASCHERE BIANCO/NERO ---
                    cv::namedWindow("2. MASCHERA FOREGROUND", cv::WINDOW_NORMAL);
                    cv::imshow("2. MASCHERA FOREGROUND", maskForeground);
                    cv::namedWindow("3. MASCHERA BACKGROUND", cv::WINDOW_NORMAL);
                    cv::imshow("3. MASCHERA BACKGROUND", maskBackground);
                    // --- FINE AGGIUNTA 3 ---

                    cv::namedWindow("7. RISULTATI DA INVIARE AL CSV", cv::WINDOW_NORMAL);
                    cv::imshow("7. RISULTATI DA INVIARE AL CSV", imgAnteprima);

                    // Aspetta che tu prema un tasto (Barra spaziatrice o Invio) per andare alla prossima immagine
                    int key = cv::waitKey(0);

                    // Se premi ESC (27)
                    if (key == 27) {
                        mostraFinestre = false; // Disattiva le finestre
                        cv::destroyAllWindows(); // Chiude quelle aperte
                        std::cout << "[INFO] Finestre chiuse! Elaborazione in corso in background per completare il CSV..." << std::endl;
                    }
                }
            }
            std::cout << "[SUCCESSO] File generato: " << dataset.outputCsv << std::endl;
        }
        std::cout << "\n[FINE GLOBALE] Pipeline Train & Test completata!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}