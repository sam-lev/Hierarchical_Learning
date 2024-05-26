import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

class Quality_Metrics:
    def __init__(self, x_ids = None,
                 y = None,
                 preds = None,
                 pred_dict = None,
                 label_dict = None,
                 cutoffs = None):
        purpose = "accuracy ect"
        self.x_ids = x_ids
        self.cutoffs = np.arange(0.01, 0.98, 0.01)
        self.preds = preds
        self.pred_dict = pred_dict
        self.label_dict = label_dict
        self.cutoffs = cutoffs
        self.y = y

        # for optimal thresholdiong of f1
        self.F1_log = {}
        self.max_f1 = 0
        self.opt_thresh = 0

        # thresholded by optimal thresh of labels and preds
        self.y_thresh = []
        self.pred_thresh = []

        # confusion matrix and true positive true negative false positive false negative
        self.true_positive = 0
        self.true_negative = 0
        self.false_positive = 0
        self.false_negative = 0
        self.confusion_matrix = [self.true_positive, self.true_negative,self.false_positive, self.false_negative]

        # final results
        self.f1_dict = {}
        self.recall_dict = {}
        self.precision_dict = {}
        self.f1_recall_precision = {}

        self.threshold_f1()
        self.confusion_matrix()
        self.compute_prediction_metrics()

    def confusion_matrix(self, x_ids = None,
                         y = None,
                         preds = None,
                         pred_dict = None,
                         label_dict = None,
                         cutoffs = None,
                         use_average =True,):


        print(" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(" METRICS ARE AVERAGED: ", use_average)
        print(" %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        x_ids = x_ids if x_ids is not None else self.x_ids
        y = y if y is not None else self.y
        preds = preds if preds is not None else self.preds
        pred_dict = pred_dict if pred_dict is not None else self.pred_dict
        label_dict = label_dict if label_dict is not None else self.label_dict

        if pred_dict is not None:
            preds = np.array(list(pred_dict.values()))
        if pred_dict is not None or label_dict is not None:
            if pred_dict is not None:
                x_ids = np.array(list(pred_dict.keys()))
            if label_dict is not None and pred_dict is None:
                x_ids = np.array(list(label_dict.keys()))
            if label_dict is not None:
                y = np.array(list(label_dict.keys()))


        cutoffs = cutoffs if cutoffs is not None else self.cutoffs

        self.cutoffs = cutoffs
        self.y = y
        self.preds = preds

        cutoff = self.opt_f1_threshold()

        for label in y:
            for infval in preds:
                # for idx, cutoff in enumerate(cutoffs):

                if infval >= cutoff:
                    if label == 1:
                        self.true_positive += 1.  # true positive
                    else:
                        self.false_positive += 1.  # false positive
                else:
                    if label == 1:
                        self.false_negative += 1.  # false negative
                    else:
                        self.true_negative += 1.  # true negative

    def threshold_f1(self, labels = None, preds = None):

        for thresh in self.cutoffs:

            threshed_labels = [logit > thresh for logit in self.y]
            threshed_preds = [logit > thresh for logit in self.preds]

            F1_score_topo = f1_score(y_true=threshed_labels,
                                     y_pred=threshed_preds, average=None)[-1]

            # self.F1_log[F1_score_topo] = thresh
            if F1_score_topo >= self.max_f1:
                self.max_f1 = F1_score_topo

                self.F1_log[self.max_f1] = thresh
                self.opt_f1_threshold = thresh

        self.y_thresh = []
        self.pred_thresh = []
        cutoff = self.opt_f1_threshold()
        for label in self.y:
            self.y_thresh.append(label >= cutoff)
        for pred in self.preds:
            self.pred_thresh.append(pred >= cutoff)
    def opt_f1_theshold(self):
        return self.F1_log[self.max_f1]

    def compute_prediction_metrics(self, predictions, labels, out_folder, threshold=None):
        threshold = self.opt_f1_threshold()
        predictions, labels = self.make_binary_prediction_pairs(predictions=predictions,
                                                                 labels=labels, threshold=threshold)
        # get dictionary of scores
        # keys are
        self.f1_dict = self.f1(predictions=predictions, labels=labels)
        self.recall_dict = self.recall(predictions=predictions, labels=labels)
        self.precision_dict = self.precision(predictions=predictions, labels=labels)

        self.f1_recall_precision = {'f1': self.f1_dict,
                                    'recall': self.recall_dict,
                                    'precision': self.precision_dict}
        # for scoring_type in f1_recall_precision.keys():
        #     write_model_scores(model=model,
        #                        scoring=scoring_type, scoring_dict=f1_recall_precision[scoring_type],
        #                        out_folder=out_folder)

    def make_binary_prediction_pairs(self, predictions, labels, threshold=0.5):
        predictions_b = self.preds
        labels_b = []
        for l in labels:
            labels_b.append(l)
        predictions_b = np.array(predictions_b)
        labels_b = np.array(self.y)
        predictions_b[predictions_b >= threshold] = 1.
        predictions_b[predictions_b < threshold] = 0.
        return predictions

    def recall(self, predictions=None, labels=None, print_score=False):
        preds = predictions
        labels = labels
        recall_micro = recall_score(y_true=labels, y_pred=preds, average="micro")
        recall_macro = recall_score(y_true=labels, y_pred=preds, average="macro")
        recall_weighted = recall_score(y_true=labels, y_pred=preds, average="weighted")
        if print_score:
            print("Recall(micro) Score ( tp / (tp +fn) ): ", recall_micro)
            print("Recall(macro) Score ( tp / (tp +fn) ): ", recall_macro)
            print("Recall(weighted) Score ( tp / (tp +fn) ): ", recall_weighted)
        score_dict = {"weighted": recall_weighted, "binary": None, "micro": recall_micro, "macro": recall_macro}
        return score_dict

    def precision(self, predictions=None, labels=None, print_score=False):
        preds = predictions
        labels = labels
        precision_micro = precision_score(y_true=labels, y_pred=preds, average="micro")
        precision_macro = precision_score(y_true=labels, y_pred=preds, average="macro")
        precision_weighted = precision_score(y_true=labels, y_pred=preds, average="weighted")
        precision_avg = average_precision_score(y_true=labels, y_score=preds)
        if print_score:
            print("Precision Score (micro): ", precision_micro)
            print("Precision Score (macro): ", precision_macro)
            print("Precision Score (weighted): ", precision_weighted)
            print("Precision Score (average): ", precision_avg)
        score_dict = {"weighted": precision_weighted, "average": precision_avg,
                      "micro": precision_micro, "macro": precision_macro}
        return score_dict

    def f1(self, predictions=None, labels=None, print_score=False):
        preds = predictions
        labels = labels
        f1_w = f1_score(labels, preds, average="weighted")
        f1_b = f1_score(labels, preds, average="binary")
        f1_mi = f1_score(labels, preds, average="micro")
        f1_ma = f1_score(labels, preds, average="macro")
        f1_class = f1_score(labels, preds, average=None)
        print("    * F1 class", f1_class)
        f1_class = f1_class[-1]
        fs = [str(f1_w), str(f1_b), str(f1_mi), str(f1_ma)]
        if print_score:
            print("F1 Score (weighted): ", f1_w)
            print("F1 Score (binary):   ", f1_b)
            print("F1 Score (micro): ", f1_mi)
            print("F1 Score (macro): ", f1_ma)
        score_dict = {"weighted": f1_w, "binary": f1_b, "micro": f1_mi, "macro": f1_ma, "class": f1_class}
        return score_dict

    def write_model_scores(self, scoring, scoring_dict, out_folder, threshold=''):
        scoring_file = os.path.join(out_folder, scoring + threshold + '.txt')
        print("... Writing scoring file to:", scoring_file)
        scoring_file = open(scoring_file, "w+")
        for mode in scoring_dict.keys():
            score = scoring_dict[mode]
            scoring_file.write(mode + ' ' + str(score) + "\n")
        scoring_file.close()