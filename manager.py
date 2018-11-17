"""
The manager allows easy terminal operation on the API
typically called: python manager.py operation [parameters]
parameters could be optional -denoted by []- or not

Examples:
    python manager.py train 64
    python manager.py finetune
    python manager.py evaluate test.jpg How many persons are there?

Allowed operations:
    train                 [batch_size]
    finetune              [batch_size]
    validate              [batch_size]
    preprocess_questions
    extract_all_features  [batch_size]
    evaluate               Image_Path Question
        Image_Path: Must not contain spaces
"""

import api
import sys

def main(args):
    batch_size = 32  # Default
    k = 10           # Default

    if len(args) >= 2:
        batch_size = int(args[1])
    if len(args) >= 3:
        k = int(args[2])
        
    if args[0] == "train":
        print("Training from scratch ...")
        print("Batch size is ", batch_size)
        api.train(batch_size, True, True, True, 100)
    elif args[0] == "finetune":
        print("Finetune checkpoint ...")
        print("Batch size is ", batch_size)
        api.train(batch_size, False, True, True, 100)
    elif args[0] == "validate":
        print("Validate The System ...")
        print("Batch size is ", batch_size)
        api.validate_system(batch_size)
    elif args[0] == "preprocess_questions":
        print("Preprocess Questions ...")
        api.prepare_data()
    elif args[0] == "extract_all_features":
        print("Extract Images Features ...")
        print("Batch size is ", batch_size)
        api.extract_features(batch_size)
    elif args[0] == "evaluate":
        print("Evaluating Example ...")
        print("Image Path: ", args[1])
        print("Question words: ", args[1:])
        api.evaluate_example_url(args[1], args[1:])
    elif args[0] == "trace":
        print("Trace Statistics of a Batch of Validation Set ...")
        print("Batch size is ", batch_size)
        print("K is ", k)
        api.trace(batch_size, k)

if __name__ == "__main__":
    main(sys.argv[1:])
