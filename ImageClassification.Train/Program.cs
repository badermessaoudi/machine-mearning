using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Common;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Vision;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    internal class Program
    {
        static void Main()
        {
            const string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
            string imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "test-images");

            // Création du MlContext
            var mlContext = new MLContext(seed: 1);

            // Paramétrage du log du context
            mlContext.Log += FilterMLContextLog;
            var fullImagesetFolderPath = Path.Combine(assetsPath, "inputs", "images", "pokemon_photos_set");

            // 1. Chargement des images source pour l'entraînement et les stocker dans un objet IDataView puis on les mélange (comme un jeu de cartes ;))
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: fullImagesetFolderPath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 2. Préparation des images et étiquettage
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion.
                    MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: fullImagesetFolderPath,
                                                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 3. Séparation des images en deux groupes : groupe d'entraînement 80% et groupe d'évaluation 20% 
            var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 4. Configuration de la pipeline du modèle et utilisation des paramètres par défaut
            //
            var pipeline = mlContext.MulticlassClassification.Trainers
                    .ImageClassification(featureColumnName: "Image",
                                         labelColumnName: "LabelAsKey",
                                         validationSet: testDataView)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                      inputColumnName: "PredictedLabel"));

            // 5. Entraînement et création du modèle
            Console.WriteLine("*** Entraînement du modèle de classification d'image ***");

            // On mesure la durée de l'entraînement
            var watch = Stopwatch.StartNew();

            //Début de l'entraînement !
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            var elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"L'entraînement a duré : {elapsedMs / 1000} seconde(s)");

            // 6. Récupération des métriques et évaluation (précision, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 7. Sauvegarde du modèle vers le répertoire d'output des assets (on obtient au final un fichier zip ML.NET du modèle et un fichier TensorFlow .pb du modèle)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Modèle sauvegardé vers : {outputMlNetModelFilePath}");

            // 8. On va faire une prédiction pour tester la consommation de notre modèle et voir sa précision
            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);

            Console.WriteLine("Appuyer sur une touche pour quitter");
            Console.ReadKey();
        }
       
        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Début des prédictions en masse pour évaluer la qualité du modèle...");

            // Measuring time
            var watch = Stopwatch.StartNew();

            var predictionsDataView = trainedModel.Transform(testDataset);

            var metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName:"LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            var elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"La prédiction et l'évaluation ont duré : {elapsed2Ms / 1000} seconde(s)");
        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            var testImages = FileUtils.LoadInMemoryImagesFromDirectory(
                imagesFolderPathForPredictions, false);

            var imageToPredict = testImages.First();

            var prediction = predictionEngine.Predict(imageToPredict);

            Console.WriteLine(
                $"Nom de l'image : [{imageToPredict.ImageFileName}], " +
                $"Scores : [{string.Join(",", prediction.Score)}], " +
                $"Label de prédiction : {prediction.PredictedLabel}");
        }


        public static IEnumerable<ImageData> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);

        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}

