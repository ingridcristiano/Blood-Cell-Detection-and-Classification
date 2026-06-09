// =============================================================================
// PROGETTO: Segmentazione e Classificazione di Cellule Ematiche
// ESAME:    Image Processing and Analysis (IPA)
// AUTORI:   Gruppo (vedi relazione)
//
// DESCRIZIONE:
//   Elabora immagini di vetrini di sangue periferico. Per ogni immagine esegue
//   la pipeline di segmentazione (WBC, RBC, PLT), estrae 25 feature numeriche
//   per segmento e le salva nei CSV per il successivo addestramento ML.
//
//   OUTPUT INTERMEDI (salvati automaticamente su disco, senza bloccare il CSV):
//     Vengono salvate 8 cartelle numerate sotto ./output_steps/, ognuna con
//     le immagini di confronto (Originale | Step) di TUTTE le immagini:
//       01_WBC_soglia_HSV/         Soglia cromatica viola grezza per i WBC
//       02_WBC_maschera_finale/    Maschera WBC dopo morfologia e filtro area
//       03_PLT_esclusione_WBC/     Zona di esclusione WBC per le piastrine
//       04_PLT_maschera_finale/    Maschera PLT dopo filtro area 30-800 px2
//       05_RBC_Otsu_foreground/    Soglia di Otsu: foreground totale
//       06_RBC_sottrazione_WBC/    Foreground - WBC dilatati = RBC grezzo
//       07_RBC_scheletro/          Scheletrizzazione morfologica della maschera RBC
//       08_RBC_finestre/           Finestre 80x80 centrate sui centri scheletro
//       09_risultato_finale/       Ground Truth (medico) vs Pipeline (nostra)
//
//   VISUALIZZAZIONE INTERATTIVA:
//     Come nel codice originale, si apre solo la finestra dell'ultimo step
//     (Ground Truth vs Pipeline). Premere un tasto per scorrere le immagini,
//     ESC per chiudere le finestre e completare il CSV in batch.
//
//   FEATURE ESTRATTE (25 per campione):
//     Geometriche (6):  Area, Perimetro, Circolarita, AspectRatio,
//                       Eccentricita, Extent
//     Colore BGR  (3):  MeanBlue, MeanGreen, MeanRed
//     Intensita   (4):  MeanValue, MinValue, MaxValue, TextureValue
//     Saturazione (4):  MeanSaturation, MinSat, MaxSat, TextureSat
//     Texture     (1):  TextureLaplacian
//     Hu Moments  (7):  Hu1..Hu7 (log-trasformati, segno preservato)
// =============================================================================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <cmath>

namespace fs = std::filesystem;


// =============================================================================
// FUNZIONE: Salva il confronto tra due immagini affiancate in una cartella step
//
// stepDir  : cartella di destinazione (es. "./output_steps/01_WBC_soglia_HSV/")
// baseName : nome del file senza estensione (es. "BloodImage_001")
// sinistra : immagine di sinistra (originale o step precedente)
// labelSin : etichetta testuale sovrapposta all'immagine sinistra
// destra   : immagine dello step corrente
// labelDes : etichetta testuale sovrapposta all'immagine destra
// =============================================================================
void salvaConfronto(const std::string& stepDir,
    const std::string& baseName,
    const cv::Mat& sinistra, const std::string& labelSin,
    const cv::Mat& destra, const std::string& labelDes) {
    fs::create_directories(stepDir);

    // Normalizziamo entrambe a BGR 3 canali
    cv::Mat s3, d3;
    if (sinistra.channels() == 1) cv::cvtColor(sinistra, s3, cv::COLOR_GRAY2BGR);
    else s3 = sinistra.clone();
    if (destra.channels() == 1)   cv::cvtColor(destra, d3, cv::COLOR_GRAY2BGR);
    else d3 = destra.clone();

    cv::resize(s3, s3, cv::Size(640, 480));
    cv::resize(d3, d3, cv::Size(640, 480));

    // Etichette descrittive in sovrimpressione
    cv::putText(s3, labelSin, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    cv::putText(d3, labelDes, cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

    cv::Mat affiancata;
    cv::hconcat(s3, d3, affiancata);

    cv::imwrite(stepDir + baseName + ".png", affiancata);
}


// =============================================================================
// FUNZIONE PRINCIPALE: Estrazione feature da una maschera binaria
//
// Per ogni contorno rilevato nella maschera (con area >= minArea) calcola
// 25 feature e scrive una riga nel CSV. Aggiorna imgAnteprima con i bbox.
//
// FEATURE:
//   [Geometriche]  area, perimetro, circolarita, aspectRatio, eccentricita, extent
//   [Colore BGR]   meanBlue, meanGreen, meanRed
//   [Intensita V]  meanValue, minValue, maxValue, textureValue (StdDev)
//   [Saturaz. S]   meanSat, minSat, maxSat, textureSat (StdDev)
//   [Texture]      textureLaplacian (StdDev del Laplaciano su canale V)
//   [Hu Moments]   hu1..hu7 (log-trasformati: sign * log10(|h|))
// =============================================================================
void extractAndSaveFeatures(const cv::Mat& imgOriginale, const cv::Mat& mask,
    const std::string& cellType, const std::string& imageName,
    std::ofstream& csvFile, cv::Mat& imgAnteprima, double minArea = 5.0) {

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Conversione in HSV per feature di intensita e saturazione
    cv::Mat imgHSV;
    cv::cvtColor(imgOriginale, imgHSV, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsvChannels;
    cv::split(imgHSV, hsvChannels);
    cv::Mat canalSaturazione = hsvChannels[1]; // Canale S
    cv::Mat canalIntensita = hsvChannels[2]; // Canale V

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < minArea) continue;

        // --- FEATURE GEOMETRICHE BASE ---
        double perimetro = cv::arcLength(contour, true);
        cv::Rect boundingBox = cv::boundingRect(contour);
        double aspectRatio = (double)boundingBox.width / (double)boundingBox.height;
        // Circolarita: 1.0 = cerchio perfetto; decresce per forme allungate o irregolari
        double circolarita = (perimetro > 0) ? (4.0 * CV_PI * area) / (perimetro * perimetro) : 0.0;

        // Eccentricita: distanza dalla forma circolare (0=cerchio, ~1=linea)
        // Calcolata dall'ellisse equivalente (richiede almeno 5 punti)
        double eccentricita = 0.0;
        if (contour.size() >= 5) {
            cv::RotatedRect ellisse = cv::fitEllipse(contour);
            double a = ellisse.size.width / 2.0; // semiasse maggiore
            double b = ellisse.size.height / 2.0; // semiasse minore
            if (a > 0 && a >= b)
                eccentricita = std::sqrt(1.0 - (b * b) / (a * a));
        }

        // Extent: rapporto area contorno / area bounding box
        // Basso per forme irregolari, alto per forme compatte
        double areaBox = (double)boundingBox.width * (double)boundingBox.height;
        double extent = (areaBox > 0) ? area / areaBox : 0.0;

        // Maschera della singola cellula (ROI per le statistiche)
        cv::Mat singleCellMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::drawContours(singleCellMask,
            std::vector<std::vector<cv::Point>>{contour},
            -1, cv::Scalar(255), cv::FILLED);

        // --- FEATURE DI COLORE (spazio BGR) ---
        cv::Scalar meanColor = cv::mean(imgOriginale, singleCellMask);

        // --- FEATURE DI INTENSITA (canale V di HSV) ---
        cv::Scalar meanInt, stdDevInt;
        cv::meanStdDev(canalIntensita, meanInt, stdDevInt, singleCellMask);
        double meanValue = meanInt[0];
        double textureValueStdDev = stdDevInt[0]; // Variabilita interna di luminosita
        double minValue, maxValue;
        cv::minMaxLoc(canalIntensita, &minValue, &maxValue, nullptr, nullptr, singleCellMask);

        // --- FEATURE DI SATURAZIONE (canale S di HSV) ---
        cv::Scalar meanSat, stdDevSat;
        cv::meanStdDev(canalSaturazione, meanSat, stdDevSat, singleCellMask);
        double meanSaturation = meanSat[0];
        double textureSatStdDev = stdDevSat[0]; // Variabilita interna di saturazione
        double minSat, maxSat;
        cv::minMaxLoc(canalSaturazione, &minSat, &maxSat, nullptr, nullptr, singleCellMask);

        // --- TEXTURE: Laplaciano su canale V ---
        // StdDev del Laplaciano = presenza di bordi interni e variazioni di textura
        cv::Mat laplacianImg;
        cv::Laplacian(canalIntensita, laplacianImg, CV_64F);
        cv::Scalar meanLap, stdDevLap;
        cv::meanStdDev(laplacianImg, meanLap, stdDevLap, singleCellMask);
        double textureLaplacian = stdDevLap[0];

        // --- MOMENTI DI HU (invarianti a traslazione, rotazione, scala) ---
        // Log-trasformazione: sign * log10(|h|) per stabilita numerica
        cv::Moments mu = cv::moments(contour, false);
        double huMoments[7];
        cv::HuMoments(mu, huMoments);
        for (int i = 0; i < 7; i++) {
            if (huMoments[i] != 0) {
                double sign = (huMoments[i] > 0) ? 1.0 : -1.0;
                huMoments[i] = sign * std::log10(std::abs(huMoments[i]));
            }
            else {
                huMoments[i] = 0.0;
            }
        }

        // --- ANNOTAZIONE VISIVA SUL FRAME DI ANTEPRIMA ---
        cv::Scalar colorBox(0, 255, 0);
        if (cellType == "GlobuloBianco") colorBox = cv::Scalar(255, 0, 0);
        else if (cellType == "GlobuloRosso")  colorBox = cv::Scalar(0, 0, 255);
        else if (cellType == "Piastrina")     colorBox = cv::Scalar(0, 255, 255);

        cv::rectangle(imgAnteprima, boundingBox, colorBox, 2);
        cv::putText(imgAnteprima, cellType.substr(0, 3),
            cv::Point(boundingBox.x, std::max(0, boundingBox.y - 5)),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, colorBox, 1);

        // --- SCRITTURA RIGA CSV (25 feature + 6 metadati) ---
        csvFile << imageName << ","
            << cellType << ","
            << boundingBox.x << "," << boundingBox.y << ","
            << boundingBox.width << "," << boundingBox.height << ","
            << area << "," << perimetro << ","
            << circolarita << "," << aspectRatio << ","
            << eccentricita << "," << extent << ","
            << meanColor[0] << "," << meanColor[1] << "," << meanColor[2] << ","
            << meanValue << "," << minValue << "," << maxValue << ","
            << meanSaturation << "," << minSat << "," << maxSat << ","
            << textureValueStdDev << "," << textureSatStdDev << ","
            << textureLaplacian << ","
            << huMoments[0] << "," << huMoments[1] << "," << huMoments[2] << ","
            << huMoments[3] << "," << huMoments[4] << "," << huMoments[5] << ","
            << huMoments[6] << "\n";
    }
}


// =============================================================================
// STRUTTURA: Configurazione di un singolo dataset (Train o Test)
// =============================================================================
struct DatasetConfig {
    std::string inputFolder;
    std::string outputCsv;
    std::string nomeFase;
};


// =============================================================================
// MAIN
// =============================================================================
int main() {
    try {
        // =====================================================================
        // CONFIGURAZIONE PERCORSI
        // NOTA: Le righe commentate sono i percorsi delle collaboratrici.
        //       Non modificare.
        // =====================================================================
      // std::string cartellaProgettoML = "C:/progetto_cellule/ProgettoML/csv/"; //giorgia
       //std::string cartellaProgettoML = "C:/Template-C-/ProgettoML/csv/";
        std::string cartellaProgettoML = "./csv/";

        fs::create_directories(cartellaProgettoML);

        // Cartella radice per gli step intermedi salvati su disco
        std::string stepRootDir = "./output_steps/";

        std::vector<DatasetConfig> pipeline = {
            //        {"C:/progetto_cellule/ProgettoIPA/archive/train/img/", cartellaProgettoML + "features_cellule_train.csv", "TRAIN"}, //giorgia

                    //{"C:/Template-C-/ProgettoIPA/archive/train/img/", cartellaProgettoML + "features_cellule_train.csv", "TRAIN"},
            { "../archive/train/img/", cartellaProgettoML + "features_cellule_train.csv", "TRAIN"},


            //            {"C:/progetto_cellule/ProgettoIPA/archive/test/img/", cartellaProgettoML + "features_cellule_test.csv", "TEST"}    //giorgia
              //      };

               //    {"C:/Template-C-/ProgettoIPA/archive/test/img/", cartellaProgettoML + "features_cellule_test.csv", "TEST"}, 
                 { "../archive/test/img/", cartellaProgettoML + "features_cellule_test.csv", "TEST" }
        };

        std::string folderAnnotate = "../output/";

        // Range colore viola globale per i globuli bianchi
        cv::Scalar lowerViolaGlobale(78, 23, 161);
        cv::Scalar upperViolaGlobale(134, 255, 252);

        // Header CSV aggiornato con 25 feature (aggiunte Eccentricita ed Extent)
        std::string csvHeader =
            "ImageName,CellType,BoxX,BoxY,BoxW,BoxH,"
            "Area,Perimeter,Circularity,AspectRatio,Eccentricity,Extent,"
            "MeanBlue,MeanGreen,MeanRed,"
            "MeanValue,MinValue,MaxValue,"
            "MeanSaturation,MinSat,MaxSat,"
            "TextureValue,TextureSat,TextureLaplacian,"
            "Hu1,Hu2,Hu3,Hu4,Hu5,Hu6,Hu7\n";


        // =====================================================================
        // CICLO SUI DATASET: TRAIN, poi TEST
        // =====================================================================
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
                std::cerr << "[WARNING] Nessuna immagine trovata in "
                    << dataset.inputFolder << ". Passo alla fase successiva." << std::endl;
                continue;
            }

            std::ofstream csvFile(dataset.outputCsv, std::ios::out | std::ios::trunc);
            if (!csvFile.is_open()) {
                std::cerr << "[ERRORE] Impossibile creare il file " << dataset.outputCsv << std::endl;
                continue;
            }
            csvFile << csvHeader;

            bool mostraFinestre = true;

            // -----------------------------------------------------------------
            // CICLO SULLE IMMAGINI
            // -----------------------------------------------------------------
            for (size_t f = 0; f < imagePaths.size(); f++) {

                cv::Mat imgOriginale = cv::imread(imagePaths[f], cv::IMREAD_COLOR);
                if (imgOriginale.empty()) continue;

                cv::Mat imgAnteprima = imgOriginale.clone();

                std::string fullPath = imagePaths[f];
                size_t lastSlash = fullPath.find_last_of("/\\");
                std::string fileName = fullPath.substr(lastSlash + 1);
                std::string baseName = fileName.substr(0, fileName.find_last_of('.'));

                std::cout << "[" << dataset.nomeFase << "] "
                    << fileName << " (" << f + 1 << "/" << imagePaths.size() << ")" << std::endl;

                cv::Mat imgAnnotataReale = cv::imread(folderAnnotate + fileName, cv::IMREAD_COLOR);
                if (imgAnnotataReale.empty()) imgAnnotataReale = imgOriginale.clone();

                // =============================================================
                // PRE-ELABORAZIONE: riduzione del rumore
                // Median (3x3): rimuove rumore impulsivo sale-e-pepe
                // Bilateral (d=9): leviga preservando i bordi delle cellule
                // =============================================================
                cv::Mat imgMedian, imgBilateral;
                cv::medianBlur(imgOriginale, imgMedian, 3);
                cv::bilateralFilter(imgMedian, imgBilateral, 9, 75, 75);

                // Conversioni colore per le pipeline successive
                cv::Mat imgHSVSub, imgGray;
                cv::cvtColor(imgBilateral, imgHSVSub, cv::COLOR_BGR2HSV);
                cv::cvtColor(imgBilateral, imgGray, cv::COLOR_BGR2GRAY);


                // =============================================================
                // SEGMENTAZIONE WBC — Globuli Bianchi
                //
                // I WBC presentano un nucleo colorato in viola/lilla dalla
                // colorazione di Wright-Giemsa. Soglia HSV nel range del viola,
                // seguita da morfologia per chiudere buchi e rimuovere rumore,
                // e filtro per area >= 1500 px2 per escludere artefatti.
                // =============================================================
                cv::Mat maskViolaGlobale;
                cv::inRange(imgHSVSub, lowerViolaGlobale, upperViolaGlobale, maskViolaGlobale);

                // STEP SALIENTE 1: la soglia HSV isola immediatamente il nucleo leucocitario
                salvaConfronto(stepRootDir + "01_WBC_soglia_HSV/",
                    baseName,
                    imgOriginale, "Originale",
                    maskViolaGlobale, "Soglia HSV viola (raw WBC)");

                cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_CLOSE,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
                cv::morphologyEx(maskViolaGlobale, maskViolaGlobale, cv::MORPH_OPEN,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));

                cv::Mat maskSoloBianchi = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                cv::Mat labelsB, statsB, centroidsB;
                int nLabelsB = cv::connectedComponentsWithStats(maskViolaGlobale,
                    labelsB, statsB, centroidsB);
                for (int i = 1; i < nLabelsB; i++) {
                    if (statsB.at<int>(i, cv::CC_STAT_AREA) >= 1500)
                        maskSoloBianchi.setTo(255, labelsB == i);
                }
                cv::morphologyEx(maskSoloBianchi, maskSoloBianchi, cv::MORPH_CLOSE,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

                // STEP SALIENTE 2: maschera WBC definitiva sovrapposta all'originale
                {
                    cv::Mat overlayWBC = imgOriginale.clone();
                    overlayWBC.setTo(cv::Scalar(255, 0, 0), maskSoloBianchi);
                    salvaConfronto(stepRootDir + "02_WBC_maschera_finale/",
                        baseName,
                        imgOriginale, "Originale",
                        overlayWBC, "WBC rilevati (blu) - area >= 1500 px2");
                }


                // =============================================================
                // SEGMENTAZIONE PLT — Piastrine
                //
                // Le piastrine appaiono come piccoli granuli blu-violacei.
                // Soglia HSV nel range del blu, seguita da esclusione esplicita
                // delle regioni WBC (evita falsi positivi nel citoplasma del
                // Globulo Bianco) e filtro dimensionale 30-800 px2.
                // =============================================================
                cv::Mat maskPiastrineRaw;
                cv::inRange(imgHSVSub,
                    cv::Scalar(85, 30, 50), cv::Scalar(150, 255, 255),
                    maskPiastrineRaw);

                cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
                cv::morphologyEx(maskPiastrineRaw, maskPiastrineRaw, cv::MORPH_CLOSE, kernelClose);

                // Costruzione zona di esclusione: bbox WBC espanso di 10 px
                cv::Mat areaDaEscludere = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                std::vector<std::vector<cv::Point>> contoursBianchi;
                cv::findContours(maskSoloBianchi, contoursBianchi,
                    cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                for (const auto& contour : contoursBianchi) {
                    cv::Rect wbcBox = cv::boundingRect(contour);
                    wbcBox.x = std::max(0, wbcBox.x - 10);
                    wbcBox.y = std::max(0, wbcBox.y - 10);
                    wbcBox.width = std::min(imgOriginale.cols - wbcBox.x, wbcBox.width + 20);
                    wbcBox.height = std::min(imgOriginale.rows - wbcBox.y, wbcBox.height + 20);
                    cv::rectangle(areaDaEscludere, wbcBox, cv::Scalar(255), cv::FILLED);
                }

                // STEP SALIENTE 3: l'esclusione WBC e' il passaggio critico per le PLT
                salvaConfronto(stepRootDir + "03_PLT_esclusione_WBC/",
                    baseName,
                    imgOriginale, "Originale",
                    areaDaEscludere, "Zona esclusa (WBC + margine 10 px)");

                cv::Mat maskSfondoLibero;
                cv::bitwise_not(areaDaEscludere, maskSfondoLibero);
                cv::bitwise_and(maskPiastrineRaw, maskSfondoLibero, maskPiastrineRaw);

                cv::Mat maskSoloPiastrine = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                cv::Mat labelsP, statsP, centroidsP;
                int nLabelsP = cv::connectedComponentsWithStats(maskPiastrineRaw,
                    labelsP, statsP, centroidsP);
                for (int i = 1; i < nLabelsP; i++) {
                    int area = statsP.at<int>(i, cv::CC_STAT_AREA);
                    if (area >= 30 && area <= 800)
                        maskSoloPiastrine.setTo(255, labelsP == i);
                }

                // STEP SALIENTE 4: piastrine finali evidenziate
                {
                    cv::Mat overlayPLT = imgOriginale.clone();
                    overlayPLT.setTo(cv::Scalar(0, 255, 255), maskSoloPiastrine);
                    salvaConfronto(stepRootDir + "04_PLT_maschera_finale/",
                        baseName,
                        imgOriginale, "Originale",
                        overlayPLT, "PLT rilevate (giallo) - 30-800 px2");
                }


                // =============================================================
                // SEGMENTAZIONE RBC — Globuli Rossi
                //
                // Strategia differenziale: i Rossi non hanno un colore
                // specifico affidabile, quindi si isolano per sottrazione.
                //
                //  1) Otsu su scala di grigi -> foreground totale
                //  2) Sottrazione WBC (con dilatazione 31x31 per includere
                //     anche il citoplasma pallido) -> maschera RBC grezza
                //  3) Opening + doppia erosione per separare Rossi sovrapposti
                //  4) Scheletrizzazione morfologica -> "filo" centrale dei cluster
                //  5) Campionamento centri lungo lo scheletro (step=40, dist.min=50)
                //  6) Finestre 80x80 centrate su ogni centro -> contorno locale
                // =============================================================

                // STEP SALIENTE 5: Otsu isola tutto il foreground
                cv::Mat maskTutteLeCellule;
                cv::threshold(imgGray, maskTutteLeCellule, 0, 255,
                    cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

                salvaConfronto(stepRootDir + "05_RBC_Otsu_foreground/",
                    baseName,
                    imgOriginale, "Originale",
                    maskTutteLeCellule, "Foreground totale (soglia Otsu)");

                // STEP SALIENTE 6: la sottrazione WBC e' il cuore della strategia RBC
                cv::Mat kernelBianchiGrande =
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(31, 31));
                cv::Mat maskBianchiDilatata;
                cv::dilate(maskSoloBianchi, maskBianchiDilatata, kernelBianchiGrande);
                cv::Mat maskRosa;
                cv::subtract(maskTutteLeCellule, maskBianchiDilatata, maskRosa);

                salvaConfronto(stepRootDir + "06_RBC_sottrazione_WBC/",
                    baseName,
                    maskTutteLeCellule, "Foreground totale",
                    maskRosa, "Foreground - WBC = RBC grezzo");

                // Opening + doppia erosione
                std::vector<std::vector<cv::Point>> contoursRosa;
                cv::findContours(maskRosa, contoursRosa, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                cv::Mat maskRosaTmp = cv::Mat::zeros(maskRosa.size(), CV_8UC1);
                cv::drawContours(maskRosaTmp, contoursRosa, -1, cv::Scalar(255), cv::FILLED);
                cv::morphologyEx(maskRosaTmp, maskRosaTmp, cv::MORPH_OPEN,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

                cv::Mat kernelErode = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                cv::Mat maskEroded;
                cv::erode(maskRosaTmp, maskEroded, kernelErode, cv::Point(-1, -1), 2);

                // Scheletrizzazione morfologica iterativa (algoritmo Zhang-Suen)
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

                // STEP SALIENTE 7: lo scheletro motiva la strategia di campionamento
                salvaConfronto(stepRootDir + "07_RBC_scheletro/",
                    baseName,
                    maskEroded, "Maschera RBC erosa",
                    skel, "Scheletro morfologico");

                // Campionamento dei centri lungo lo scheletro
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
                    for (const auto& c : listaCentri) {
                        if (cv::norm(pt - c) < distanzaMinima) { troppoVicino = true; break; }
                    }
                    if (!troppoVicino) listaCentri.push_back(pt);
                }

                // STEP SALIENTE 8: finestre 80x80 centrate sui punti campionati
                {
                    cv::Mat imgFinestre = imgOriginale.clone();
                    for (const auto& pt : listaCentri) {
                        int x = std::max(0, pt.x - latoRettangolo / 2);
                        int y = std::max(0, pt.y - latoRettangolo / 2);
                        int w = std::min(imgOriginale.cols - x, latoRettangolo);
                        int h = std::min(imgOriginale.rows - y, latoRettangolo);
                        cv::rectangle(imgFinestre, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 1);
                    }
                    salvaConfronto(stepRootDir + "08_RBC_finestre/",
                        baseName,
                        imgOriginale, "Originale",
                        imgFinestre, "Finestre 80x80 px sui centri RBC");
                }


                // =============================================================
                // ESTRAZIONE FEATURE E SCRITTURA CSV
                // =============================================================
                extractAndSaveFeatures(imgOriginale, maskSoloBianchi,
                    "GlobuloBianco", fileName, csvFile, imgAnteprima, 1500.0);
                extractAndSaveFeatures(imgOriginale, maskSoloPiastrine,
                    "Piastrina", fileName, csvFile, imgAnteprima);

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
                    cv::findContours(porzioneRossoReale, localContours,
                        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    for (const auto& localContour : localContours) {
                        double localArea = cv::contourArea(localContour);
                        if (localArea >= 300.0) {
                            cv::Mat maskStrettaDefinitiva = cv::Mat::zeros(imgOriginale.size(), CV_8UC1);
                            cv::drawContours(maskStrettaDefinitiva,
                                std::vector<std::vector<cv::Point>>{localContour},
                                -1, cv::Scalar(255), cv::FILLED);
                            extractAndSaveFeatures(imgOriginale, maskStrettaDefinitiva,
                                "GlobuloRosso", fileName,
                                csvFile, imgAnteprima, 300.0);
                        }
                    }
                }

                // =============================================================
                // RISULTATO FINALE — Salvataggio e visualizzazione interattiva
                //
                // Salva sempre su disco il confronto Ground Truth vs Pipeline.
                // La finestra interattiva si comporta come nell'originale:
                //   - Qualsiasi tasto: prossima immagine
                //   - ESC: chiude le finestre, prosegue in batch
                // =============================================================
                salvaConfronto(stepRootDir + "09_risultato_finale/",
                    baseName,
                    imgAnnotataReale, "Ground Truth (medico)",
                    imgAnteprima, "Pipeline IPA (nostra)");

                if (mostraFinestre) {
                    cv::putText(imgAnteprima, dataset.nomeFase,
                        cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                        1, cv::Scalar(0, 0, 0), 2);
                    cv::namedWindow("1. GUIDA REALE", cv::WINDOW_NORMAL);
                    cv::namedWindow("7. RISULTATI DA INVIARE AL CSV", cv::WINDOW_NORMAL);
                    cv::imshow("1. GUIDA REALE", imgAnnotataReale);
                    cv::imshow("7. RISULTATI DA INVIARE AL CSV", imgAnteprima);

                    int key = cv::waitKey(0);
                    if (key == 27) {
                        mostraFinestre = false;
                        cv::destroyAllWindows();
                        std::cout << "[INFO] Finestre chiuse! Elaborazione veloce in background per completare il CSV..." << std::endl;
                    }
                }

            } // fine ciclo immagini

            std::cout << "[SUCCESSO] File generato con 25 Feature: " << dataset.outputCsv << std::endl;
        }

        std::cout << "\n[FINE GLOBALE] Pipeline Train & Test completata con successo!" << std::endl;
        std::cout << "Step intermedi salvati in: " << stepRootDir << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Errore a runtime: " << e.what() << std::endl;
    }

    return 0;
}