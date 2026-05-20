#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>

namespace fs = std::filesystem;

void extractAndSaveFeatures(const cv::Mat& imgOriginale, const cv::Mat& mask,
    const std::string& cellType, const std::string& imageName,
    std::ofstream& csvFile) {

    std::vector<std::vector<cv::Point>> contours;
    // Troviamo i contorni esterni nella maschera
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);

        // Filtriamo rumore microscopico (puoi aggiustare questo valore)
        if (area < 5.0) continue;

        double perimetro = cv::arcLength(contour, true);

        // Bounding box per calcolare l'Aspect Ratio
        cv::Rect boundingBox = cv::boundingRect(contour);
        double aspectRatio = (double)boundingBox.width / (double)boundingBox.height;

        // Circolarità: 4 * pi * Area / (Perimetro^2). Più si avvicina a 1, più è un cerchio perfetto.
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

        std::string folderOriginali = "C:\\Users\\giorg\\OneDrive\\Desktop\\progetto_m_l_1\\example_images\\";
        std::string folderAnnotate = "C:\\Users\\giorg\\OneDrive\\Desktop\\progetto_m_l_1\\output\\";

        std::string outFolderBianchi = "C:\\Users\\giorg\\OneDrive\\Desktop\\progetto_m_l_1\\output_bianchi\\";
        std::string outFolderPiastrine = "C:\\Users\\giorg\\OneDrive\\Desktop\\progetto_m_l_1\\output_piastrine\\";
        std::string outFolderRossi = "C:\\Users\\giorg\\OneDrive\\Desktop\\progetto_m_l_1\\output_rossi\\";

        	/*	std::string folderOriginali = "C:\\Template-C-\\example_images\\";
			std::string folderAnnotate = "C:\\Template-C-\\output\\";

			std::string outFolderBianchi = "C:\\Template-C-\\output_bianchi\\";
			std::string outFolderPiastrine = "C:\\Template-C-\\output_piastrine\\";
			std::string outFolderRossi = "C:\\Template-C-\\output_rossi\\";*/

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

        // Parametri per i globuli bianchi (dal primo codice)
        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // ... (il tuo codice di setup delle cartelle) ...

        // SETUP FILE CSV PER LE FEATURE
        std::string csvPath = "C:\\Users\\giorg\\OneDrive\\Desktop\\progetto_m_l_1\\features_cellule.csv";
        std::ofstream csvFile(csvPath);
        if (!csvFile.is_open()) {
            std::cerr << "ERRORE: Impossibile creare il file CSV." << std::endl;
            return -1;
        }
        // Scriviamo l'header del CSV
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

            // --- PRE-PROCESSING PER BIANCHI E ROSSI (Dal Codice 1) ---
            cv::Mat imgMedian, imgBilateral;
            cv::medianBlur(imgOriginale, imgMedian, 3);
            cv::bilateralFilter(imgMedian, imgBilateral, 9, 75, 75);

            cv::Mat imgHSV, imgGray;
            cv::cvtColor(imgBilateral, imgHSV, cv::COLOR_BGR2HSV);
            cv::cvtColor(imgBilateral, imgGray, cv::COLOR_BGR2GRAY);

            // =====================================================================
            // 1. SEZIONE: GLOBULI BIANCHI (Dal Codice 1)
            // =====================================================================
            cv::Mat maskViolaGlobale;
            cv::inRange(imgHSV, lowerViolaGlobale, upperViolaGlobale, maskViolaGlobale);

            cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

            cv::Mat maskSoloBianchi = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
            cv::Mat labelsB, statsB, centroidsB;
            int nLabelsB = cv::connectedComponentsWithStats(maskViolaGlobale, labelsB, statsB, centroidsB);

            // Filtraggio area per i globuli bianchi
            for (int i = 1; i < nLabelsB; i++) {
                if (statsB.at<int>(i, cv::CC_STAT_AREA) >= 800) {
                    maskSoloBianchi.setTo(255, labelsB == i);
                }
            }
            // Rifinitura finale globulo bianco
            cv::morphologyEx(maskSoloBianchi, maskSoloBianchi, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));


            // =====================================================================
            // 2. SEZIONE: PIASTRINE (Dal Codice 2 - Metodo Black-Hat)
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

            // Filtraggio area perfetta per piastrine
            for (int i = 1; i < nLabelsP; i++) {
                int area = statsP.at<int>(i, cv::CC_STAT_AREA);
                if (area >= 6 && area <= 300) {
                    maskSoloPiastrine.setTo(255, labelsP == i);
                }
            }

            cv::Mat maskPiastrineVis;
            cv::dilate(maskSoloPiastrine, maskPiastrineVis, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));


            // =====================================================================
            // 3. SEZIONE: GLOBULI ROSSI (Metodo Skeleton -> Solo Centri)
            // =====================================================================
            cv::Mat maskTutteLeCellule;
            cv::threshold(imgGray, maskTutteLeCellule, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

            // Riempimento dei buchi
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(maskTutteLeCellule, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::Mat maskRosa = cv::Mat::zeros(maskTutteLeCellule.size(), CV_8UC1);
            cv::drawContours(maskRosa, contours, -1, cv::Scalar(255), cv::FILLED);
            cv::morphologyEx(maskRosa, maskRosa, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

            // 1. Erosione preventiva (fondamentale per lo skeleton per ridurre le ramificazioni)
            cv::Mat kernelErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
            cv::Mat maskEroded;
            cv::erode(maskRosa, maskEroded, kernelErode, cv::Point(-1, -1), 3); // 3 iterazioni per staccare i grappoli

            // 2. Calcolo dello Skeleton
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

            // 3. Estrazione e Campionamento dei punti
            std::vector<cv::Point> skelPoints;
            cv::findNonZero(skel, skelPoints);

            // Creiamo una maschera nera dove inseriremo SOLO i puntini
            cv::Mat maskSoloCentri = cv::Mat::zeros(maskRosa.size(), CV_8UC1);

            // --- PARAMETRO CHIAVE ---
            // Lo 'step' decide ogni quanti pixel dello skeleton prendiamo un punto.
            // Se vedi una "fila" di troppi punti sulla stessa cellula, AUMENTA questo valore (es. a 30 o 40).
            int step = 60;

            // Creiamo anche una lista per salvare le coordinate (ti servirà per il Machine Learning)
            std::vector<cv::Point> listaCentri;

            for (size_t i = 0; i < skelPoints.size(); i += step) {
                listaCentri.push_back(skelPoints[i]);

                // Disegniamo un puntino corposo (raggio 4) per vederlo bene nella Dashboard
                cv::circle(maskSoloCentri, skelPoints[i], 8, cv::Scalar(255), cv::FILLED);
            }

            // Riassegna la maschera per visualizzarla su "5. MASK ROSSI"
            maskRosa = maskSoloCentri.clone();



            // =====================================================================
            // SALVATAGGIO
            // =====================================================================
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);
            cv::imwrite(outFolderRossi + fileName, maskRosa);

            // ... (il tuo codice di salvataggio delle maschere) ...
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);
            cv::imwrite(outFolderRossi + fileName, maskRosa);

            // =====================================================================
            // ESTRAZIONE FEATURE
            // =====================================================================
            extractAndSaveFeatures(imgOriginale, maskSoloBianchi, "GlobuloBianco", fileName, csvFile);
            extractAndSaveFeatures(imgOriginale, maskSoloPiastrine, "Piastrina", fileName, csvFile);
            extractAndSaveFeatures(imgOriginale, maskRosa, "GlobuloRosso", fileName, csvFile);



            // =====================================================================
            // DASHBOARD
            // =====================================================================
            cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
            cv::imshow("1. GUIDA REALE", imgAnnotataReale);

            cv::namedWindow("3. MASK BIANCHI", cv::WINDOW_NORMAL);
            cv::imshow("3. MASK BIANCHI", maskSoloBianchi);

            cv::namedWindow("4. MASK PIASTRINE", cv::WINDOW_NORMAL);
            cv::imshow("4. MASK PIASTRINE", maskPiastrineVis);

            cv::namedWindow("5. MASK ROSSI", cv::WINDOW_NORMAL);
            cv::imshow("5. MASK ROSSI", maskRosa);

            int key = cv::waitKey(0);
            if (key == 27) break; // ESC per uscire
        }
        std::cout << "\n[FINE] Elaborazione completata con sezioni separate!" << std::endl;
    }// Fine del ciclo for

         catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}