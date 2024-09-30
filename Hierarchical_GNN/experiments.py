from Hierarchical_GNN.GraphSampling.HierGNN_Experiments import HierSGNN

class succesive_comparison():
    # fp = open('./run_logs/memory_profiler.log', 'w+')
    # @profile#stream=fp)
    def __init__(self, args):
        self.HST_model = HierSGNN(args,
                                  to_do= "ADD ARGUMENTS PASSED TO Hierarchical Succesive",
                                  experimental = True)
        sequential_schemes = ("successive_init", "successive_lift")
        for sequential_type in sequential_schemes:
            self.HST_model = HierSGNN(args,
                                      to_do="ADD ARGUMENTS PASSED TO Hierarchical Succesive",
                                      experimental=True,
                                      sequential_type=sequential_type)
        #           # inference test results
        #           # if non-sequential:
        #               # x_i -> 0 before during graph lift
        #               # begin training from epoch 0 to N/(number graphs)
        #           # if seqeuntial
        #               # initialize living nodes with learned node embedding
        #               # continue training

        #       ## record final result per sublevel graph in graph filtration sequence
        #
        #     ## learned filtrations of graph, learned graph importance
        #       ## for hierarchical joint learning training and inference
        #           # at epoch N/number graphs
        #               # for graph in filtration hierarchy
        #                   # test accuracy
        #       ## avg_filt_val_graph = 0
        #       ## for each node and respective learned filtration value
        #
        #
        #
