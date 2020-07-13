using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using ConsoleApp1.Model;
using Microsoft.VisualBasic;

namespace ConsoleApp1
{
    class Program
    {
        static void Main(string[] args)
        {
            string rootDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "../../../"));
            //string dbFilePath = Path.Combine(rootDir, "Data", "DailyDemand.mdf");
            string modelPath = Path.Combine(rootDir, "MLModel.zip");
            //            var connectionString = $"Data Source=(LocalDB)\\MSSQLLocalDB;AttachDbFilename={dbFilePath};Integrated Security=True;Connect Timeout=30;";
            
            // you can use TblSmartAr.sql and able to make this table
            var connectionString = $"Data Source = (localdb)\\MSSQLLocalDB; Initial Catalog = Ar_Database; Integrated Security = True; Connect Timeout = 30; Encrypt = False; TrustServerCertificate = False; ApplicationIntent = ReadWrite; MultiSubnetFailover = False;";

            
            MLContext mlContext = new MLContext();

            DatabaseLoader loader = mlContext.Data.CreateDatabaseLoader<ModelInput>();

            string query = "SELECT Invoice_Date, CAST(Due_year as REAL) as Due_year, CAST(OverDue_Days as REAL) as OverDue_Days FROM TblSmartAR";

            DatabaseSource dbSource = new DatabaseSource(SqlClientFactory.Instance,
                                            connectionString,
                                            query);
            DateTime now = DateTime.Today;

            int thisyear = (now.Year);

            
            IDataView dataView = loader.Load(dbSource);

            IDataView firstYearData = mlContext.Data.FilterRowsByColumn(dataView, "Due_year", upperBound: thisyear);
            IDataView secondYearData = mlContext.Data.FilterRowsByColumn(dataView, "Due_year", lowerBound: thisyear);

            var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
                outputColumnName: "ForecastedDays",
                inputColumnName: "OverDue_Days",
                windowSize: 80,
                seriesLength: 90,
                trainSize: 365,
                horizon: 80,
                confidenceLevel: .90f,
                confidenceLowerBoundColumn: "MinimumDays",
                confidenceUpperBoundColumn: "MaximumDays");

            SsaForecastingTransformer forecaster = forecastingPipeline.Fit(firstYearData);

            Evaluate(secondYearData, forecaster, mlContext);

            var forecastEngine = forecaster.CreateTimeSeriesEngine<ModelInput, ModelOutput>(mlContext);
            forecastEngine.CheckPoint(mlContext, modelPath);
            //foreach(var j in )
            //{
            //    if()
            //}          
            
            Forecast(secondYearData, 40, forecastEngine, mlContext);

            Console.ReadKey();
        }

        static void Evaluate(IDataView testData, ITransformer model, MLContext mlContext)
        {
            // Make predictions
            IDataView predictions = model.Transform(testData);

            // Actual values
            IEnumerable<float> actual =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, true)
                    .Select(observed => observed.OverDue_Days);

            // Predicted values
            IEnumerable<float> forecast =
                mlContext.Data.CreateEnumerable<ModelOutput>(predictions, true)
                    .Select(prediction => prediction.ForecastedDays[0]);

            // Calculate error (actual - forecast)
            var metrics = actual.Zip(forecast, (actualValue, forecastValue) => actualValue - forecastValue);

            // Get metric averages
            var MAE = metrics.Average(error => Math.Abs(error)); // Mean Absolute Error
            var RMSE = Math.Sqrt(metrics.Average(error => Math.Pow(error, 2))); // Root Mean Squared Error

            // Output metrics
            Console.WriteLine("Evaluation Metrics");
            Console.WriteLine("---------------------");

            Console.WriteLine($"Mean Absolute Error: {MAE:F3}");
            Console.WriteLine($"Root Mean Squared Error: {RMSE:F3}\n");
        }

        static void Forecast(IDataView testData, int horizon, TimeSeriesPredictionEngine<ModelInput, ModelOutput> forecaster, MLContext mlContext)
        {

            ModelOutput forecast = forecaster.Predict();
            int i = 0;

            IEnumerable<string> forecastOutput =
                mlContext.Data.CreateEnumerable<ModelInput>(testData, reuseRowObject: false)
                    .Take(horizon)
                    .Select((ModelInput rental, int index) =>
                    {
                        i++;
                       
                        string Date = rental.Invoice_Date.ToShortDateString();
                        float actualDays = rental.OverDue_Days;
                        float lowerEstimate = Math.Max(0, forecast.MinimumDays[index]);
                        float estimate = forecast.ForecastedDays[index];
                        float upperEstimate = forecast.MaximumDays[index];
                        return
                      //  $"No                    : {i}\n" +
                        $"Date                  : {Date}\n" +
                        $"Actual Days           : {actualDays}\n" +
                        $"Lower Estimate Days   : {lowerEstimate}\n" +
                        $"Forecast Days         : {estimate}\n" +
                        $"Upper Estimate Days   : {upperEstimate}\n";
                    });

            // Output predictions
            Console.WriteLine("Rental Forecast");
            Console.WriteLine("---------------------");
            foreach (var prediction in forecastOutput)
            {
                Console.WriteLine(prediction);
            }

        }
    }
}
