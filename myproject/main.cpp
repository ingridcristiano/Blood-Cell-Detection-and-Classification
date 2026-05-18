#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    try {
        // =========================================================================
        // 1. SETUP PERCORSI PARALLELI (ORIGINALI VS ANNOTATE)
        // =========================================================================
        std::string folderOriginali = "C:/Progetti/Template C++/example_images/";
        std::string folderAnnotate = "C:/Progetti/Template C++/output/";

        std::string outFolderBianchi = "C:/Progetti/Template C++/output_bianchi/";
        std::string outFolderPiastrine = "C:/Progetti/Template C++/output_piastrine/";
        std::string outFolderRossi = "C:/Progetti/Template C++/output_rossi/";

        fs::create_directories(outFolderBianchi);
        fs::create_directories(outFolderPiastrine);
        fs::create_directories(outFolderRossi);

        std::vector<cv::String> imagePaths;
        cv::glob(folderOriginali + "*.jpeg", imagePaths);

        if (imagePaths.empty()) {
            std::cerr << "ERRORE: Nessuna immagine trovata in example_images." << std::endl;
            return -1;
        }

        // I NOSTRI PARAMETRI STORICI PER LA MASCHERA UNICA VIOLA
        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // =========================================================================
        // 2. CICLO DI ELABORAZIONE
        // =========================================================================
        for (size_t f = 0; f < imagePaths.size(); f++) {
            cv::Mat imgOriginale = cv::imread(imagePaths[f], cv::IMREAD_COLOR);
            if (imgOriginale.empty()) continue;

            std::string fullPath = imagePaths[f];
            size_t lastSlash = fullPath.find_last_of("/\\");
            std::string fileName = fullPath.substr(lastSlash + 1);

            cv::Mat imgAnnotataReale = cv::imread(folderAnnotate + fileName, cv::IMREAD_COLOR);
            if (imgAnnotataReale.empty()) imgAnnotataReale = imgOriginale.clone();

            

            // =====================================================================
            // FASE A: PRE-PROCESSING (BILATERAL FILTER - NIENTE CLAHE)
            // =====================================================================
            // 1. Un leggerissimo median blur per togliere la polvere (sale e pepe) senza intaccare i bordi
            cv::Mat imgMedian;
            cv::medianBlur(imgOriginale, imgMedian, 3);

            // 2. Bilateral Filter: Sfoca il plasma ma mantiene i bordi delle cellule taglienti!
            cv::Mat imgBilateral;
            cv::bilateralFilter(imgMedian, imgBilateral, 9, 75, 75);

            // 3. Conversioni standard senza manipolazioni del contrasto
            cv::Mat imgHSV, imgGray;
            cv::cvtColor(imgBilateral, imgHSV, cv::COLOR_BGR2HSV);
            cv::cvtColor(imgBilateral, imgGray, cv::COLOR_BGR2GRAY);

            // =====================================================================
            // FASE B: LA MASCHERA UNICA VIOLA E LO SMISTAMENTO PER AREA
            // =====================================================================
            cv::Mat maskViolaGlobale;
            cv::inRange(imgHSV, lowerViolaGlobale, upperViolaGlobale, maskViolaGlobale);

            cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
            cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

            cv::Mat maskSoloBianchi = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
            cv::Mat maskSoloPiastrineRaw = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);

            cv::Mat labels, stats, centroids;
            int nLabels = cv::connectedComponentsWithStats(maskViolaGlobale, labels, stats, centroids);

            for (int i = 1; i < nLabels; i++) {
                int area = stats.at<int>(i, cv::CC_STAT_AREA);

                if (area >= 800) {
                    maskSoloBianchi.setTo(255, labels == i);
                }
                else if (area >= 35 && area <= 250) {
                    maskSoloPiastrineRaw.setTo(255, labels == i);
                }
            }

            // =====================================================================
            // FASE C: RIFINITURA BIANCHI E PIASTRINE (Gestione Alone)
            // =====================================================================
           
            cv::morphologyEx(maskSoloBianchi, maskSoloBianchi, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5)));

			//dilatiamo i bianchi per creare una zona di esclusione intorno a loro, così da non confondere le piastrine vicine con il citoplasma del bianco
            cv::Mat zonaEsclusioneCitoplasma;
            cv::dilate(maskSoloBianchi, zonaEsclusioneCitoplasma, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(35, 35)));

            cv::Mat maskSoloPiastrine;
            cv::bitwise_and(maskSoloPiastrineRaw, ~zonaEsclusioneCitoplasma, maskSoloPiastrine);

            // MODIFICATO: Portiamo il Size da (5, 5) a (9, 9) per spazzare via i 3 grumi falsi lontani dal bianco
            cv::morphologyEx(maskSoloPiastrine, maskSoloPiastrine, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));

            cv::Mat maskPiastrineVis;
            // MODIFICATO: Portiamo da (5, 5) a (7, 7) per rigonfiare bene la piastrina vera superstite
            cv::dilate(maskSoloPiastrine, maskPiastrineVis, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

            // =====================================================================
            // FASE D: GLOBULI ROSSI (OTSU)
            // =====================================================================
            cv::Mat maskTutteLeCellule;
            cv::threshold(imgGray, maskTutteLeCellule, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(maskTutteLeCellule, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::Mat maskCellulePiene = cv::Mat::zeros(maskTutteLeCellule.size(), CV_8UC1);
            cv::drawContours(maskCellulePiene, contours, -1, cv::Scalar(255), cv::FILLED);

            cv::Mat maskRosa;
            cv::subtract(maskCellulePiene, zonaEsclusioneCitoplasma, maskRosa);
            cv::subtract(maskRosa, maskSoloPiastrine, maskRosa);

            cv::morphologyEx(maskRosa, maskRosa, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));

            // =====================================================================
            // FASE E: SALVATAGGIO
            // =====================================================================
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);
            cv::imwrite(outFolderRossi + fileName, maskRosa);

            // =====================================================================
            // FASE F: LA DASHBOARD A 5 FINESTRE
            // =====================================================================
            cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
            cv::imshow("1. GUIDA REALE", imgAnnotataReale);

            cv::namedWindow("3. MASK BIANCHI", cv::WINDOW_NORMAL);
            cv::imshow("3. MASK BIANCHI", maskSoloBianchi);

            cv::namedWindow("4. MASK PIASTRINE", cv::WINDOW_NORMAL);
            cv::imshow("4. MASK PIASTRINE", maskPiastrineVis);

            cv::namedWindow("5. MASK ROSSI (Otsu)", cv::WINDOW_NORMAL);
            cv::imshow("5. MASK ROSSI (Otsu)", maskRosa);

            int key = cv::waitKey(0);
            if (key == 27) break; // ESC esce dal ciclo
        }
        std::cout << "\n[FINE] Elaborazione completata!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}