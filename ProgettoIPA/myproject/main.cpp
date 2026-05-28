#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>

namespace fs = std::filesystem;

// Funzione potenziata per estrarre le feature dal contorno (29 feature totali)
void extractAndSaveFeatures(const cv::Mat& imgOriginale, const cv::Mat& mask,
    const std::string& cellType, const std::string& imageName,
    std::ofstream& csvFile, cv::Mat& imgAnteprima, double minArea = 5.0) {

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Conversione in HSV ed estrazione dei canali per intensità e saturazione (consigliato dal Prof)
    cv::Mat imgHSV;
    cv::cvtColor(imgOriginale, imgHSV, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(imgHSV, hsvChannels);
    cv::Mat canalSaturazione = hsvChannels[1]; // Canale S (Saturation)
    cv::Mat canalIntensita = hsvChannels[2];   // Canale V (Value / Intensità)

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < minArea) continue;

        // 1. FEATURE GEOMETRICHE BASE
        double perimetro = cv::arcLength(contour, true);
        cv::Rect boundingBox = cv::boundingRect(contour);
        double aspectRatio = (double)boundingBox.width / (double)boundingBox.height;
        double circolarita = (perimetro > 0) ? (4.0 * CV_PI * area) / (perimetro * perimetro) : 0.0;

        // Maschera della singola cellula corrente
        cv::Mat singleCellMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(singleCellMask, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);

        // 2. FEATURE DI COLORE STANDARD (BGR)
        cv::Scalar meanColor = cv::mean(imgOriginale, singleCellMask);

        // 3. INTENSITÀ LUMINOSA (Canale V di HSV)
        cv::Scalar meanInt, stdDevInt;
        cv::meanStdDev(canalIntensita, meanInt, stdDevInt, singleCellMask);
        double meanValue = meanInt[0];
        double textureValueStdDev = stdDevInt[0]; // Texture basata sulla variazione di luminosità

        double minValue, maxValue;
        cv::minMaxLoc(canalIntensita, &minValue, &maxValue, nullptr, nullptr, singleCellMask);

        // 4. PUREZZA DEL COLORE (Canale S di HSV)
        cv::Scalar meanSat, stdDevSat;
        cv::meanStdDev(canalSaturazione, meanSat, stdDevSat, singleCellMask);
        double meanSaturation = meanSat[0];
        double textureSatStdDev = stdDevSat[0]; // Texture basata sulla variazione di purezza del colore

        double minSat, maxSat;
        cv::minMaxLoc(canalSaturazione, &minSat, &maxSat, nullptr, nullptr, singleCellMask);

        // 5. TEXTURE AVANZATA (Laplaciano su Intensità HSV)
        cv::Mat laplacianImg;
        cv::Laplacian(canalIntensita, laplacianImg, CV_64F);
        cv::Scalar meanLap, stdDevLap;
        cv::meanStdDev(laplacianImg, meanLap, stdDevLap, singleCellMask);
        double textureLaplacian = stdDevLap[0]; // Rugosità/presenza di bordi netti interni

        // 6. MOMENTI DI HU (BPP) - Invarianti a rotazione e scala
        cv::Moments mu = cv::moments(contour, false);
        double huMoments[7];
        cv::HuMoments(mu, huMoments);

        // Log-trasformazione per stabilizzare i valori matematici per il Machine Learning
       
        for (int i = 0; i < 7; i++) {
            if (huMoments[i] != 0) {
                // Calcoliamo il segno: 1 se positivo, -1 se negativo
                double sign = (huMoments[i] > 0) ? 1.0 : -1.0;
                huMoments[i] = -1.0 * sign * std::log10(std::abs(huMoments[i]));
            }
            else {
                huMoments[i] = 0.0;
            }
        }
        // 7. ANNOTAZIONE VISIVA SUL VETRINO
        cv::Scalar colorBox(0, 255, 0);
        if (cellType == "GlobuloBianco") colorBox = cv::Scalar(255, 0, 0);
        else if (cellType == "GlobuloRosso") colorBox = cv::Scalar(0, 0, 255);
        else if (cellType == "Piastrina") colorBox = cv::Scalar(0, 255, 255);

        cv::rectangle(imgAnteprima, boundingBox, colorBox, 2);
        cv::putText(imgAnteprima, cellType.substr(0, 3), cv::Point(boundingBox.x, std::max(0, boundingBox.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, colorBox, 1);

        // 8. SCRITTURA COMPLETA DI TUTTE LE 29 FEATURE SUL CSV
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
            << meanColor[0] << "," // MeanBlue
            << meanColor[1] << "," // MeanGreen
            << meanColor[2] << "," // MeanRed
            << meanValue << ","    // Intensità Media (V)
            << minValue << ","     // Intensità Minima (V)
            << maxValue << ","     // Intensità Massima (V)
            << meanSaturation << ","// Saturazione Media (S)
            << minSat << ","       // Saturazione Minima (S)
            << maxSat << ","       // Saturazione Massima (S)
            << textureValueStdDev << "," // Texture della Luminosità
            << textureSatStdDev << ","   // Texture della Saturazione
            << textureLaplacian << ","   // Rugosità dei dettagli interni
            << huMoments[0] << "," << huMoments[1] << "," << huMoments[2] << ","
            << huMoments[3] << "," << huMoments[4] << "," << huMoments[5] << ","
            << huMoments[6] << "\n";
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
        // 1. CONFIGURAZIONE DEI PERCORSI ASSOLUTI
        // =========================================================================
        std::string cartellaProgettoML = "C:/progetto_cellule/ProgettoML/csv/"; //giorgia

        // Creiamo la directory di output
        fs::create_directories(cartellaProgettoML);

        std::vector<DatasetConfig> pipeline = {
           {"C:/progetto_cellule/ProgettoIPA/archive/train/img/", cartellaProgettoML + "features_cellule_train.csv", "TRAIN"}, //giorgia
           {"C:/progetto_cellule/ProgettoIPA/archive/test/img/", cartellaProgettoML + "features_cellule_test.csv", "TEST"}    //giorgia
        };

        std::string folderAnnotate = "../output/";

        // Range colore viola globale per i globuli bianchi
        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // Stringa Header aggiornata con i nomi di tutte le 29 colonne
        std::string csvHeader = "ImageName,CellType,BoxX,BoxY,BoxW,BoxH,Area,Perimeter,Circularity,AspectRatio,MeanBlue,MeanGreen,MeanRed,MeanValue,MinValue,MaxValue,MeanSaturation,MinSat,MaxSat,TextureValue,TextureSat,TextureLaplacian,Hu1,Hu2,Hu3,Hu4,Hu5,Hu6,Hu7\n";

        // =========================================================================
        // 2. CICLO SUI DATASET (Train e Test)
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

            // APERTURA FILE CSV PRINCIPALE
            std::ofstream csvFile(dataset.outputCsv, std::ios::out | std::ios::trunc);
            if (!csvFile.is_open()) {
                std::cerr << "[ERRORE] Impossibile creare il file " << dataset.outputCsv << std::endl;
                continue;
            }
            csvFile << csvHeader; // Scrive l'intestazione aggiornata

            // CREAZIONE DEL FILE CSV PER IL CLASSIFICATORE BINARIO
            std::string outputCsvBinario = dataset.outputCsv;
            size_t posCsv = outputCsvBinario.find(".csv");
            if (posCsv != std::string::npos) outputCsvBinario.insert(posCsv, "_BINARIO");

            std::ofstream csvBinario(outputCsvBinario, std::ios::out | std::ios::trunc);
            if (csvBinario.is_open()) {
                csvBinario << csvHeader; // Scrive l'intestazione aggiornata anche qui
            }

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

                cv::Mat imgHSVSub, imgGray;
                cv::cvtColor(imgBilateral, imgHSVSub, cv::COLOR_BGR2HSV);
                cv::cvtColor(imgBilateral, imgGray, cv::COLOR_BGR2GRAY);

                // ESTRAZIONE MASCHERE FOREGROUND E BACKGROUND
                cv::Mat maskAdattiva, maskForeground, maskBackground;
                cv::adaptiveThreshold(imgGray, maskAdattiva, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 101, 5);

                maskForeground = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                maskBackground = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);

                std::vector<std::vector<cv::Point>> contoursBin;
                cv::findContours(maskAdattiva, contoursBin, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                for (size_t i = 0; i < contoursBin.size(); i++) {
                    double areaBin = cv::contourArea(contoursBin[i]);
                    if (areaBin >= 150.0) {
                        cv::drawContours(maskForeground, contoursBin, (int)i, cv::Scalar(255), cv::FILLED);
                    }
                    else if (areaBin >= 10.0 && areaBin < 150.0) {
                        cv::drawContours(maskBackground, contoursBin, (int)i, cv::Scalar(255), cv::FILLED);
                    }
                }

                if (csvBinario.is_open()) {
                    cv::Mat anteprimaFantasma = imgOriginale.clone();
                    extractAndSaveFeatures(imgOriginale, maskForeground, "Foreground", fileName, csvBinario, anteprimaFantasma, 150.0);
                    extractAndSaveFeatures(imgOriginale, maskBackground, "Background", fileName, csvBinario, anteprimaFantasma, 10.0);
                }

                // --- BIANCHI ---
                cv::Mat maskViolaGlobale;
                cv::inRange(imgHSVSub, lowerViolaGlobale, upperViolaGlobale, maskViolaGlobale);
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

                // --- ESTRAZIONE FINALE COINVOLGENDO LE NUOVE FEATURE ---
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

                // GESTIONE INTERFACCIA VISIVA
                if (mostraFinestre) {
                    cv::putText(imgAnteprima, dataset.nomeFase, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
                    cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
                    cv::imshow("1. GUIDA REALE", imgAnnotataReale);
                    cv::namedWindow("2. MASCHERA FOREGROUND", cv::WINDOW_NORMAL);
                    cv::imshow("2. MASCHERA FOREGROUND", maskForeground);
                    cv::namedWindow("3. MASCHERA BACKGROUND", cv::WINDOW_NORMAL);
                    cv::imshow("3. MASCHERA BACKGROUND", maskBackground);
                    cv::namedWindow("7. RISULTATI DA INVIARE AL CSV", cv::WINDOW_NORMAL);
                    cv::imshow("7. RISULTATI DA INVIARE AL CSV", imgAnteprima);

                    int key = cv::waitKey(0);

                    if (key == 27) { // Se premi ESC
                        mostraFinestre = false;
                        cv::destroyAllWindows();
                        std::cout << "[INFO] Finestre chiuse! Elaborazione veloce in background per completare il CSV..." << std::endl;
                    }
                }
            }
            std::cout << "[SUCCESSO] File generato con 29 Feature: " << dataset.outputCsv << std::endl;
        }
        std::cout << "\n[FINE GLOBALE] Pipeline Train & Test completata con successo!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}