'''
args.py

For changing the arguments for the model
'''
import argparse

def parameter_parser():
    
    parser = argparse.ArgumentParser(description="PartGNN")
    
    parser.add_argument("--verbose",
                        default=True,
                        action='store_false',
                        help="Prints debug info")
    
    parser.add_argument("--dataset",
                        nargs="?",
                        default="./Dataset/abc_dataset",
                        help="Folder with training dataset.")


    parser.add_argument("--epochs",
                        type=int,
                        default=2,
                        help="Number of training epochs. Default is 10.")
                        
    parser.add_argument("--trainplot",
                        default=True,
                        action="store_false",
                        help="Create a training plot during training")
                        
    parser.add_argument("--print_freq",
                        type=int,
                        default=1,
                        help="How often to print information about the training process, Default=1")

    parser.add_argument("--trained_model",
                        nargs="?",
                        default="trained_model",
                        help="File name for trained model")

    parser.add_argument("--model_loc",
                        nargs="?",
                        default='.\\saved_models\\partfind_v2_simdata_3layers_16_NNConv_40epoch.pt',
                        help="Location of trained model.")




    # parser.add_argument("--GNN-1",
                        # type=int,
                        # default=128,
                        # help="Size of 1st GNN. Default is 128.")

    # parser.add_argument("--GNN-2",
                        # type=int,
                        # default=64,
                        # help="Size of 2nd GNN. Default is 64.")

    # parser.add_argument("--GNN-3",
                        # type=int,
                        # default=32,
                        # help="Size of 3rd GNN. Default is 32.")

    # parser.add_argument("--tensor-neurons",
                        # type=int,
                        # default=16,
                        # help="Neurons in tensor network layer. Default is 16.") ########

    # parser.add_argument("--bottle-neck-neurons",
                        # type=int,
                        # default=16,
	                # help="Bottle neck layer neurons. Default is 16.")

    # parser.add_argument("--batch-size",
                        # type=int,
                        # default=1,
                        # help="Number of graph pairs per batch. Default is 128.")

    # parser.add_argument("--dropout",
                        # type=float,
                        # default=0.5,
                        # help="Dropout rate. Default is 0.5.")

    # parser.add_argument("--learning-rate",
                        # type=float,
                        # default=0.001,
                        # help="Learning rate. Default is 0.001.")

    # parser.add_argument("--weight-decay",
                        # type=float,
                        # default=5*10**-4,
                        # help="Adam weight decay. Default is 5*10^-4.")

    # parser.add_argument("--histogram",
                        # dest="histogram",
                        # action="store_true")

    # parser.set_defaults(histogram=False)
    
    parser.add_argument("--dataset-range",
                        nargs='+',
                        type=int,
                        help="Set dataset compiling range.")

    return parser.parse_args()