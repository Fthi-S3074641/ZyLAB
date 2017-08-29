using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FthiAbadi
{
    class Program
    {
        private static List<string> emotions = new List<string>() { "anger", "fear", "joy", "sadness", "surprise" };
        private static List<string> sentiment = new List<string>() { "positive", "negative" };

        static void Main(string[] args)
        {

            string dest = ".../Dataset/";
            string DatasetLanguage = "EN";
            Prepare DataP = new Prepare(dest, DatasetLanguage + "_senti_train.txt", DatasetLanguage + "_senti_test.txt");
            //DataP.Translate(dest, "NL_emotion_test.txt");
            string DictionaryLocation = ".../NRC-Emotion-Lexicon-v0.92/";
            string embeddingLocation = ".../embeddings/" + DatasetLanguage + ".txt";
            Detector EmoDetection = new Detector(DictionaryLocation, DatasetLanguage, embeddingLocation, DataP);
            //EmoDetection.CheckVocublary(embeddingLocation); // to check if the word list has a corrsponding Word2Vec
            EmoDetection.DoClassification(DataP, sentiment);
        }
    }
}
