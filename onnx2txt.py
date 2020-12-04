import onnx

model_path = "roi_pool.onnx"


def main():
    # Load ONNX File
    model = onnx.load(model_path)

    # print all nodes in a model
    print("====== Nodes ======")
    for i, node in enumerate(model.graph.node):
        print("[Node #{}]".format(i))
        print(node)

    # print input data of model
    print("====== Inputs ======")
    for i, input in enumerate(model.graph.input):
        print("[Input #{}]".format(i))
        print(input)

    # print output data of model
    print("====== Outputs ======")
    for i, output in enumerate(model.graph.output):
        print("[Output #{}]".format(i))
        print(output)


if __name__ == "__main__":
    main()