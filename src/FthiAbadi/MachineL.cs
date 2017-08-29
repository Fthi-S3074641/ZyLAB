using System;
using System.Collections.Generic;
using Accord.IO;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics;
using Accord.MachineLearning;
using System.IO;
using System.Linq;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Kernels;
using Word2vec.Tools;

public class MachineL
{
    public string W2Vlocation;
    public static Vocabulary W2Vvocublary;
    public static SupportVectorMachine<Gaussian> KernelSVM;
    public static SequentialMinimalOptimization<Gaussian> Kernels;
    public static LinearCoordinateDescent teacher;
    public static SupportVectorMachine svm;
    public static TFIDF codeB;
    public static string language;
    private static List<string> Nomiss;
    public MachineL( string fpath, string langg, List<string[]> LearTF)
	{
        language = langg;
        W2Vlocation = fpath;
        W2Vvocublary = new Word2VecTextReader().Read(W2Vlocation);
        Kernels = new SequentialMinimalOptimization<Gaussian>()
        {
            UseComplexityHeuristic = true,
            //Complexity = 100,
            UseKernelEstimation = true,

        };
        codeB = new TFIDF()
        {
            Tf = TermFrequency.Log,
            Idf = InverseDocumentFrequency.Default
        };
        //codeB.Learn(LearTF.ToArray());
        //Console.WriteLine(codeB.NumberOfWords);
        Console.WriteLine("w2v is read");
        Nomiss = new List<string>();
    }
    public string[] Getmissed()
    {
        return Nomiss.ToArray();
    }
    public void TrainSession(List<string[]> Trainset, int[] booleanint, string emo)
    {
        double[][] Tset = Embedder(Trainset);// W2Vectorizer() can be used as alternative as well
        //double[][] Tset = codeB.Transform(Trainset.ToArray());
        string filename = Path.Combine(".../ZyLAB_Trained", language + "_" + emo + "_EmoKernel.accord");
        KernelSVM = Kernels.Learn(Tset, booleanint);
        Serializer.Save(obj: KernelSVM, path: filename);
        Console.WriteLine("Training is Done!!");
    }

    public int[] TestSession(List<string[]> TestData, string emo)
    {
        double[][] TestD = Embedder(TestData);// W2Vectorizer() can be used as alternative as well
        //double[][] TestD = codeB.Transform(TestData.ToArray());
        string filename = Path.Combine(".../ZyLAB_Trained", language+"_" + emo + "_EmoKernel.accord");
        KernelSVM = Serializer.Load<SupportVectorMachine<Gaussian>>(filename);
        bool[] answers = KernelSVM.Decide(TestD);
        int[] zeroOneAnswers = answers.ToZeroOne();
        return zeroOneAnswers;
    }

    // This function can be used instead of the W2Vectorizer if you are confident there are no empty sentences
    public double[][] Embedder(List<string[]> Datas)
    {
        List<double[]> reslt = new List<double[]>();
        foreach (string[] sar in Datas)
        {
            //try
            //{
                double[] VectorRep = Array.ConvertAll(W2Vvocublary.GetSummRepresentationOrNullForPhrase(sar).NumericVector, x => (double)x);
                reslt.Add(VectorRep);
        //}
        //    catch
        //{
        //    Console.WriteLine(string.Join(" ", sar));
        //}

    }
        return reslt.ToArray();
    }

    public double[][] W2Vectorizer(List<string[]> DataC)
    {
        List<double[]> reslt = new List<double[]>();
       
        foreach (string[] s in DataC)
        {
            List<double> sente = new List<double>();
            List<double[]> wrv = new List<double[]>();
            List<double> idfs = new List<double>();
            
            foreach (string w in s)
            {

                if (W2Vvocublary.ContainsWord(w))
                {
                    double[] doubleArray = Array.ConvertAll(W2Vvocublary[w].NumericVector, x => (double)x);
                    wrv.Add(doubleArray);
                }
                else
                {
                    if (!Nomiss.Contains(w))
                    {
                        Nomiss.Add(w);
                    }
                    
                }
            }

            for (int i = 0; i < W2Vvocublary.VectorDimensionsCount; i++)
            {
                if (wrv.Count >= 1)
                {
                    double avg = 0;
                    for (int j = 0; j < wrv.Count; j++)// double[] wrd in wrv)
                    {
                        double ss = (double)wrv[j][i];
                        avg = (double)wrv[j][i] + (double)avg;
                    }
                    sente.Add(avg);
                }
                else
                {
                    sente.Add((double)0);
                }
            }
            reslt.Add(sente.ToArray());
        }
       // Console.WriteLine("Words missed = " + Nomiss.Count());
        return reslt.ToArray();
    }
}
