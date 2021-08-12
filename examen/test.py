from detectron2.engine import default_argument_parser
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(args, type(args))