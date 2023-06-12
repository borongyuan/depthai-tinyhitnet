#include <depthai/depthai.hpp>

cv::Mat fromPlanarFp16(const std::vector<float> &data, int w, int h)
{
    cv::Mat frame = cv::Mat(h, w, CV_8UC1);
    for (int i = 0; i < w * h; i++)
        frame.data[i] = (uint8_t)(data.data()[i] / 192 * 255);
    return frame;
}

int main(int argc, char **argv)
{
    auto nnPath = std::string(argv[1]);
    std::cout << "Using blob at path: " << nnPath.c_str() << std::endl;

    dai::Pipeline pipeline;

    auto monoLeft = pipeline.create<dai::node::MonoCamera>();
    auto monoRight = pipeline.create<dai::node::MonoCamera>();
    auto stereo = pipeline.create<dai::node::StereoDepth>();
    auto manipLeft = pipeline.create<dai::node::ImageManip>();
    auto manipRight = pipeline.create<dai::node::ImageManip>();
    auto tinyHITNet = pipeline.create<dai::node::NeuralNetwork>();

    auto xoutNN = pipeline.create<dai::node::XLinkOut>();

    xoutNN->setStreamName("disp");

    monoLeft->setBoardSocket(dai::CameraBoardSocket::LEFT);
    monoLeft->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoLeft->setFps(2);
    monoRight->setBoardSocket(dai::CameraBoardSocket::RIGHT);
    monoRight->setResolution(dai::MonoCameraProperties::SensorResolution::THE_800_P);
    monoRight->setFps(2);

    stereo->setDepthAlign(dai::StereoDepthProperties::DepthAlign::RECTIFIED_LEFT);

    manipLeft->initialConfig.setResize(320, 200);
    manipRight->initialConfig.setResize(320, 200);

    tinyHITNet->setBlobPath(nnPath);
    tinyHITNet->setNumInferenceThreads(1);
    tinyHITNet->setNumNCEPerInferenceThread(2);

    monoLeft->out.link(stereo->left);
    monoRight->out.link(stereo->right);
    stereo->rectifiedLeft.link(manipLeft->inputImage);
    stereo->rectifiedRight.link(manipRight->inputImage);
    manipLeft->out.link(tinyHITNet->inputs["left"]);
    manipRight->out.link(tinyHITNet->inputs["right"]);
    tinyHITNet->out.link(xoutNN->input);

    dai::Device device(pipeline);

    auto dispQueue = device.getOutputQueue("disp", 8, false);

    device.setIrLaserDotProjectorBrightness(0);
    device.setIrFloodLightBrightness(200);

    while (true)
    {
        auto disp = dispQueue->get<dai::NNData>();
        cv::Mat disparity;
        cv::resize(fromPlanarFp16(disp->getFirstLayerFp16(), 320, 200), disparity, cv::Size(640, 400));
        cv::applyColorMap(disparity, disparity, cv::COLORMAP_TURBO);
        cv::imshow("Disparity", disparity);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q')
            return 0;
    }

    return 0;
}