#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    try {
        // =========================================================================
        // 0. SETUP PERCORSI
        // =========================================================================
        std::string folderOriginali = "C:/Progetti/Template C++/example_images/";
        std::string folderAnnotate = "C:/Progetti/Template C++/output/";

        std::string outFolderBianchi = "C:/Progetti/Template C++/output_bianchi/";
        std::string outFolderPiastrine = "C:/Progetti/Template C++/output_piastrine/";
        std::string outFolderRossi = "C:/Progetti/Template C++/output_rossi/";

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
            // 3. SEZIONE: GLOBULI ROSSI (Dal Codice 1 - Metodo OTSU)
            // =====================================================================
            cv::Mat maskTutteLeCellule;
            cv::threshold(imgGray, maskTutteLeCellule, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(maskTutteLeCellule, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::Mat maskCellulePiene = cv::Mat::zeros(maskTutteLeCellule.size(), CV_8UC1);
            cv::drawContours(maskCellulePiene, contours, -1, cv::Scalar(255), cv::FILLED);

            // Assegniamo semplicemente la maschera trovata senza fare sottrazioni
            cv::Mat maskRosa = maskCellulePiene.clone();

            cv::morphologyEx(maskRosa, maskRosa, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)));


            // =====================================================================
            // SALVATAGGIO
            // =====================================================================
            cv::imwrite(outFolderBianchi + fileName, maskSoloBianchi);
            cv::imwrite(outFolderPiastrine + fileName, maskSoloPiastrine);
            cv::imwrite(outFolderRossi + fileName, maskRosa);


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
    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }
    return 0;
}