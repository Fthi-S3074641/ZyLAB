using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System.Net;
using System.Text.RegularExpressions;
using OpenNLP.Tools.SentenceDetect;

public class Prepare
{
    private static List<string> emotions = new List<string>() { "anger", "fear", "joy", "sadness", "surprise" };
    private static List<string> sentiment = new List<string>() { "positive", "negative" };
    public static Dictionary<string, string>  TrainData;
    private static Dictionary<string, string> TestData;
    private static Dictionary<string, string> Corpus;

    public Prepare(string fpath, string train, string test)
	{
        TrainData = new Dictionary<string, string>();
        TestData = new Dictionary<string, string>();
        Corpus = new Dictionary<string, string>();
        TrainData = ReadData(fpath + train);
        Console.WriteLine("Training size" + TrainData.Count());
        TestData = ReadData(fpath + test);
        Console.WriteLine("Test size" + TestData.Count());
    }
    public Dictionary<string, string> ReadData(string flocation)
    {
        Dictionary<string, string> DataX = new Dictionary<string, string>();
        StreamReader srnone = new StreamReader(flocation);
        string sstrline = "";
        string[] _valuess = null;
        while (!srnone.EndOfStream)
        {
            sstrline = srnone.ReadLine();
            _valuess = sstrline.Split('\t');
            if (_valuess.Length == 2)
            {
                string label = _valuess[0].ToLower();
                if (label=="none" || emotions.Contains(label) || sentiment.Contains(label))
                {
                    string  sentence = _valuess[1].ToLower();
                    if (!DataX.ContainsKey(sentence))
                    {
                        DataX.Add(sentence, label);
                        Corpus.Add(sentence, label);
                    }
                }
            }
        }
        srnone.Close();
        return DataX;
    }

    public List<string> TFCorpus()
    {
        return Corpus.Keys.ToList();
    }
    public Dictionary<string, string> GetTraining()
    {
        return TrainData;
    }
    public Dictionary<string, string> GetTesting()
    {
        return TestData;
    }
    public List<int> GetExpected(string indv, Dictionary<string, string> Sour)
    {
        List<int> label = new List<int>();
        int size = 0;
        foreach (string s in Sour.Values)
        {
            if (s == indv)
            {
                label.Add(1);
                size++;

            }
            else
            {
                label.Add(0);
            }
        }
        Console.WriteLine(indv + " ------------ " + size + " / " + label.Count());
        return label;
    }

    public void Translate(string fileLok, string fnm)
    {
        //var modelPath = "C:/Users/FthiA/Desktop/Emotion extraction/Phase11/OpenNlp-master/OpenNlp-master/Resources/Models/EnglishSD.nbin";
        //EnglishMaximumEntropySentenceDetector sentenceDetector = new EnglishMaximumEntropySentenceDetector(modelPath);
        List<string> Dset = new List<string>();
        StreamWriter sw = new StreamWriter(fileLok + fnm);
        foreach (var kg in TestData)// string s in DataSamples)
        {
            //List<string> newsentences = new List<string>();
            //var sentences = sentenceDetector.SentenceDetect(DataSamples[j]);
            //foreach (var h in sentences)
            //{
                string newL = TranslateText(kg.Key, "en" + "|" + "nl");
            //    newsentences.Add(newL);
            //}
            //string last = string.Join(" ", newsentences.ToArray());
            sw.WriteLine(kg.Value + "\t" + newL);
            Dset.Add(newL);
        }
        Console.WriteLine(Dset.Count() + " " + TestData.Count());
        sw.Close();
    }


    public static string TranslateText(string input, string languagePair)
    {
        string url = String.Format("http://www.google.com/translate_t?hl=en&ie=UTF8&text={0}&langpair={1}", input, languagePair);
        WebClient webClient = new WebClient();
        webClient.Encoding = System.Text.Encoding.UTF8;
        string FinalCov = "";
        try
        {
            string result = webClient.DownloadString(url);
            Regex rg1 = new Regex(@"id=result_box(.*)</");
            Match match1 = rg1.Match(result);
            string str1 = match1.Groups[1].Value;
            Regex rg2 = new Regex(@"><(.*)</");
            Match match2 = rg2.Match(str1);
            string str2 = match2.Groups[1].Value;
            Regex rg3 = new Regex(@">(.*)</");
            Match match3 = rg3.Match(str2);
            string str3 = match3.Groups[1].Value;
            FinalCov = str3.Substring(0, str3.IndexOf("</"));
        }
        catch
        {
            FinalCov = "";
        }
        return FinalCov;
    }

}
