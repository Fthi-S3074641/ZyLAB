using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Iveonik.Stemmers;
using OpenNLP.Tools.SentenceDetect;
using OpenNLP.Tools.Tokenize;
using Accord.Statistics.Analysis;
using Word2vec.Tools;

public class Detector
{
    private static List<string> emotions = new List<string>() { "anger", "fear", "joy", "sadness", "surprise" };
    private static List<string> sentiment = new List<string>() { "positive", "negative" };
    private static IStemmer StemDown;
    private static EnglishRuleBasedTokenizer tokenizer;
    private static EnglishMaximumEntropySentenceDetector sentenceDetector;
    private static Dictionary<string, string> Stopdict;
    private static Dictionary<string, Dictionary<string, string>> dictwords;
    private static Dictionary<string, List<string>> Detected;
    private static Dictionary<string, string> Translation;
    private static List<string> CorpusEmo;
    public static MachineL MLkernel;


    public Detector(string DictionaryLocation, string lann, string w2v, Prepare DataSrc)
	{
        // download the OpenNLp dotNet implementation from Github https://github.com/AlexPoint/OpenNlp
        var modelPath = ".../OpenNlp-master/OpenNlp-master/Resources/Models/EnglishSD.nbin";
        sentenceDetector = new EnglishMaximumEntropySentenceDetector(modelPath);
        tokenizer = new EnglishRuleBasedTokenizer(false); //the false is for the split on hyphen
        string[] Stopwords = File.ReadAllLines(@".../Dataset/stopwords.txt");
        var hash = new HashSet<string>(Stopwords);
        Stopdict = hash.ToArray().ToDictionary(key => key, value => value);
        StemDown = new EnglishStemmer();
        //StemDown = new DutchStemmer();
        ReadEmotionDictionary(DictionaryLocation, lann);
        MLkernel = new MachineL(w2v, lann, CorpusTokenize(DataSrc.TFCorpus()));
    }

    public void ReadEmotionDictionary(string loc, string lann)
    {
        dictwords = new Dictionary<string, Dictionary<string, string>>();
        Translation = new Dictionary<string, string>();
        foreach (string lab in emotions)
        {
            dictwords.Add(lab, new Dictionary<string, string>());
        }
        List<string> DictionaryTranslation = System.IO.File.ReadAllLines(loc + lann+"_translation.txt").ToList();
        int DicSize = 0;
        foreach (string line in DictionaryTranslation)
        {
            string[] _val = line.Split(',');
            if (_val.Length == 2 && _val[1].Any())
            {
                DicSize++;
                Translation.Add(_val[0], _val[1]);
            }
        }
        StreamReader srr = new StreamReader(loc + "wordlist.txt");
        string strline2 = "";
        string[] _values2 = null;
        while (!srr.EndOfStream)
        {
            strline2 = srr.ReadLine();
            _values2 = strline2.Split('\t');
            if (emotions.Contains(_values2[1]) && int.Parse(_values2[2]) == 1)
            {
                if (Translation.ContainsKey(_values2[0]))
                {
                    string Keyword = Translation[_values2[0]];
                    string stemmr = StemDown.Stem(Keyword);
                    if (!dictwords[_values2[1]].ContainsKey(Keyword))
                    {
                        dictwords[_values2[1]].Add(Keyword, stemmr);
                    }
                }
            }
        }
        srr.Close();
        Console.WriteLine("Our dictionary is build from "+dictwords.Count()+" emotion categories");
    }

    public void CheckVocublary(string W2Vlocation)
    {
        Vocabulary W2Vvocublary = new Word2VecTextReader().Read(W2Vlocation);
        foreach(var ky in dictwords)
        {
            Console.WriteLine("-------- "+ky.Key+" --------");
            int i = 0;
            foreach(string kyw in ky.Value.Keys)
            {
                if (!W2Vvocublary.ContainsWord(kyw))
                {
                    Console.WriteLine(i++ +kyw);
                }
            }
        }
    }
    public static List<string> MYDictionary(List<string> TokenizedSentence)
    {
        CorpusEmo = new List<string>();
        Detected = new Dictionary<string, List<string>>();
        List<string> winstr = new List<string>();
        foreach (var dict in dictwords)
        {
            int hit = 0;
            Detected.Add(dict.Key, new List<string>());
            foreach (string w in TokenizedSentence)
            {

                if (dict.Value.ContainsKey(w) || dict.Value.ContainsValue(StemDown.Stem(w)))
                {
                    Detected[dict.Key].Add(w);
                    hit++;
                    if (!CorpusEmo.Contains(w))
                    {
                        CorpusEmo.Add(w);
                    }
                }
            }
            if (hit >= 1)
            {
                winstr.Add(dict.Key);
            }
        }
        return winstr;
    }

    public List<string[]> CorpusTokenize(List<string> corpus)
    {
        List<string[]> newCorpus = new List<string[]>();
        foreach (var Text in corpus)
        {
            var tokens = tokenizer.Tokenize(Text);
            List<string> newtokens = new List<string>();
            foreach (string k in tokens)
            {
                if (k.Length >= 2 && !Stopdict.ContainsKey(k))//
                {
                    newtokens.Add(k);
                }
            }
                newCorpus.Add(newtokens.ToArray());
            
        }
        return newCorpus;
    }

    public List<string> SplitSentence(string Text)
    {
        List<string> newsentences = new List<string>();
        var sentences = sentenceDetector.SentenceDetect(Text);
        foreach (var h in sentences)
        {
            if (h.Length >= 3)
            {
                newsentences.Add(h);
            }
        }
        return newsentences;
    }

    public double EmoLevel(List<string> WinEmos, string target)
    {
        double scor = 0;

        string[] Stopwords = Detected[target].ToArray();
        var hash = new HashSet<string>(Stopwords);
        List<string> tokenEmos = hash.ToList();
        scor = (double)Detected[target].Count() / (double)CorpusEmo.Count();
        return scor;
    }
    public bool EmoWinner(List<string> WinEmos, string target)
    {
        bool scor = false;
        foreach (string emo in WinEmos)
        {
            if (emo == target)
            {
                continue;
            }
            if (Detected[emo].Count() <= Detected[target].Count())
            {
                scor = true;
            }
            else
            {
                scor = false;
            }
        }
        return scor;
    }
    public bool EmoSingularity(List<string> emos, string target)
    {
        bool alone = true;

        emos.Remove(target);
        if (emos.Any())
        {
            foreach (string tok in Detected[target])
            {
                alone = true;
                foreach (string inp in emos)
                {
                    if (Detected[inp].Contains(tok))
                    {
                        alone = false;
                    }
                }
                if (alone)
                {
                    break;
                }
            }
        }
        else
        {
            alone = true;
        }
        return alone;
    }

    public List<int> PredictEM(List<string[]> Src, string target)
    {
        List<double> pro = new List<double>();
        List<int> predicted = new List<int>();
        for (int i = 0; i < Src.Count(); i++)// string[] sentence in Src)
        {
            List<string> WinEmos = MYDictionary(Src[i].ToList());
            if (WinEmos.Contains(target))//&& EmoLevel(WinEmos,target) >=1
            {
                if (EmoSingularity(WinEmos, target))
                {
                    predicted.Add(1);
                    pro.Add(1);
                }
                else
                {
                    double db = EmoLevel(WinEmos, target);
                    if (EmoWinner(WinEmos, target) || (db >= (double)1 / (double)3 && db <= (double)2 / (double)3))//&& EmoWinner(WinEmos, target)
                    {
                        predicted.Add(1);
                        pro.Add(db);
                    }
                    else
                    {
                        predicted.Add(0);
                        pro.Add(0);
                    }
                }
            }
            else
            {
                predicted.Add(0);
                pro.Add(0);
            }
        }
        return predicted;
    }

    public List<double> ComputePR(List<int> sings, int[] decision)
    {
        List<double> retpr = new List<double>();
        int positiveValue = 1;
        int negativeValue = 0;
        int[] sins = sings.ToArray();
        ConfusionMatrix matrix = new ConfusionMatrix(decision, sins, positiveValue, negativeValue);
        Console.WriteLine("{0} {1} {2} {3} {4} {5} {6}", matrix.TruePositives, matrix.FalsePositives, matrix.TrueNegatives, matrix.FalseNegatives, matrix.Precision, matrix.Recall, matrix.FScore);
        retpr.Add(matrix.Precision);
        retpr.Add(matrix.Recall);
        retpr.Add(matrix.FScore);
        return retpr;
    }
    public void PredictEmotions(Dictionary<string, string> DataS, string emotion)
    {
        List<string[]> Samples = CorpusTokenize(DataS.Keys.ToList());
        List<int> ex = GetExpected(emotion, DataS);
        int[] Pred = PredictEM(Samples, emotion).ToArray();
        ComputePR(ex, Pred);
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
    public List<string[]> FilterEmotionWords(List<string> corpus)
    {
        List<string[]> Input = CorpusTokenize(corpus);
        List<string[]> NewCp = new List<string[]>();
        foreach (string[] toks in Input)
        {
            List<string> obj = new List<string>();
            foreach (var dict in dictwords)
            {
                foreach (string k in toks)
                {
                    if (dict.Value.ContainsKey(k) || dict.Value.ContainsValue(StemDown.Stem(k)))
                    {
                        obj.Add(k);
                    }
                }
            }
            
            //if (obj.Any())
            //{
                NewCp.Add(obj.ToArray());
            //}
            //else
            //{
            //    NewCp.Add(toks);
            //}
            
        }
        return NewCp;
    }

    public void DoClassification(Prepare Pdata, List<string> emos)
    {
        // Use the function FilterEmotionWords() instead of CorpusTokenize() to try the filtered emotion words version of the project

        //training phase
        List<string[]> TrainingSet = CorpusTokenize(Pdata.GetTraining().Keys.ToList());
        foreach (string v in emos)
        {
            MLkernel.TrainSession(TrainingSet, Pdata.GetExpected(v, Pdata.GetTraining()).ToArray(), v);
        }
        // classification phase
        List<string[]> TestSet = CorpusTokenize(Pdata.GetTesting().Keys.ToList());
        foreach (string v in emos)
        {
            List<int> ex = Pdata.GetExpected(v, Pdata.GetTesting());
            int[] pred = MLkernel.TestSession(TestSet, v);
            ComputePR(ex, pred);
        }
        System.IO.File.WriteAllLines("C:/Users/FthiA/Desktop/Emotion extraction/Phase12/Dataset/Missed_words.txt", MLkernel.Getmissed());
    }

}
