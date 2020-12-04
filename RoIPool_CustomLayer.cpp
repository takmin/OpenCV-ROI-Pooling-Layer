/****************************************************
* RoI Pooling of OpenCV DNN Layer
* Main code of RoIPoolLayer::forward() is based on pytorch:
* https://github.com/pytorch/vision/blob/master/torchvision/csrc/cpu/ROIPool_cpu.cpp
*****************************************************/

#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/layer.details.hpp>  // CV_DNN_REGISTER_LAYER_CLASS
#include <iostream>



class RoIPoolLayer : public cv::dnn::Layer
{
public:
    RoIPoolLayer(const cv::dnn::LayerParams& params) : Layer(params)
    {
        spatial_scale = params.get<float>("spatial_scale");
        cv::dnn::DictValue pooled_shape = params.get("pooled_shape");
        pooled_height = pooled_shape.getIntValue(0);
        pooled_width = pooled_shape.getIntValue(1);
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new RoIPoolLayer(params));
    }

    /*
    * inputs[0] shape of input image tensor
    * inputs[1] shape of roi box
    */
    virtual bool getMemoryShapes(const std::vector<std::vector<int> >& inputs,
        const int requiredOutputs,
        std::vector<std::vector<int> >& outputs,
        std::vector<std::vector<int> >& internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs); CV_UNUSED(internals);
        std::vector<int> outShape(4);
        outShape[0] = inputs[1][0];  // number of box
        outShape[1] = inputs[0][1];  // number of channels
        outShape[2] = this->pooled_height;
        outShape[3] = this->pooled_width;
        outputs.assign(1, outShape);
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        cv::Mat& inp = inputs[0];
        cv::Mat& box = inputs[1];
        cv::Mat& out = outputs[0];
        const float* input = (float*)inp.data;
        const float* rois = (float*)box.data;
        float* output = (float*)out.data;
        const int num_rois = box.size[0];
        const int batchSize = inp.size[0];
        const int channels = inp.size[1];
        const int height = inp.size[2];
        const int width = inp.size[3];

        for (int n = 0; n < num_rois; ++n) {
            const float* offset_rois = rois + n * 5;
            int roi_batch_ind = offset_rois[0];
            int roi_start_w = round(offset_rois[1] * spatial_scale);
            int roi_start_h = round(offset_rois[2] * spatial_scale);
            int roi_end_w = round(offset_rois[3] * spatial_scale);
            int roi_end_h = round(offset_rois[4] * spatial_scale);

            // Force malformed ROIs to be 1x1
            int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
            float bin_size_h = static_cast<float>(roi_height) / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width) / static_cast<float>(pooled_width);

            for (int ph = 0; ph < pooled_height; ++ph) {
                for (int pw = 0; pw < pooled_width; ++pw) {
                    int hstart = static_cast<int>(floor(static_cast<float>(ph) * bin_size_h));
                    int wstart = static_cast<int>(floor(static_cast<float>(pw) * bin_size_w));
                    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1) * bin_size_h));
                    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1) * bin_size_w));

                    // Add roi offsets and clip to input boundaries
                    hstart = std::min(std::max(hstart + roi_start_h, 0), height);
                    hend = std::min(std::max(hend + roi_start_h, 0), height);
                    wstart = std::min(std::max(wstart + roi_start_w, 0), width);
                    wend = std::min(std::max(wend + roi_start_w, 0), width);
                    bool is_empty = (hend <= hstart) || (wend <= wstart);

                    for (int c = 0; c < channels; ++c) {
                        // Define an empty pooling region to be zero
                        float maxval = is_empty ? 0 : -FLT_MAX;
                        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
                        int maxidx = -1;

                        const float* input_offset =
                            input + (roi_batch_ind * channels + c) * height * width;

                        for (int h = hstart; h < hend; ++h) {
                            for (int w = wstart; w < wend; ++w) {
                                int input_index = h * width + w;
                                if (input_offset[input_index] > maxval) {
                                    maxval = input_offset[input_index];
                                    maxidx = input_index;
                                }
                            }
                        }
                        int index =
                            ((n * channels + c) * pooled_height + ph) * pooled_width + pw;
                        output[index] = maxval;
                    } // channels
                } // pooled_width
            } // pooled_height
        } // num_rois
    }
private:
    int pooled_width, pooled_height;
    float spatial_scale;
};


int main(int argc, char* argv[])
{
    try {
        // Register RoIPoolLayer as MaxRoiPool
        CV_DNN_REGISTER_LAYER_CLASS(MaxRoiPool, RoIPoolLayer);

        // Load ONNX file
        cv::dnn::Net net = cv::dnn::readNet("roi_pool.onnx");

        // Load Image file
        cv::Mat img = cv::imread("2007_000720.jpg");

        // Image to tensor
        cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255);

        // ROI
        cv::Mat rois(1,5,CV_32FC1);
        rois.at<float>(0, 0) = 0;
        rois.at<float>(0, 1) = 216;
        rois.at<float>(0, 2) = 112;
        rois.at<float>(0, 3) = 304;
        rois.at<float>(0, 4) = 267;

        // set inputs
        net.setInput(blob,"input"); // don't forget name for multiple inputs
        net.setInput(rois, "boxes");

        // Forward
        cv::Mat output = net.forward(); // output.dims == 4

        // print results
        std::cout << "size: " << output.size[0] << " x " << output.size[1] << " x " << output.size[2] << " x " << output.size[3] << std::endl;
        int offset = 0;
        for (int ch = 0; ch < output.size[1]; ch++) {
            for (int r = 0; r < output.size[2]; r++) {
                for (int c = 0; c < output.size[3]; c++) {
                    std::cout << *((float*)output.data + offset) << ", ";
                    offset++;
                }
                std::cout << std::endl;;
            }
            std::cout << std::endl;;
        }
    }
    catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
}
