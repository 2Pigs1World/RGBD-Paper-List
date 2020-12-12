# Details

##### iDFusion: Globally Consistent Dense 3D Reconstruction from RGB-D and Inertial Measurements  

###### [[pdf](https://dl.acm.org/doi/abs/10.1145/3343031.3351085)] ACM MM 2020

> We present a practical fast, globally consistent and robust dense 3D reconstruction system, iDFusion, by exploring the joint benefit of both the visual (RGB-D) solution and inertial measurement unit (IMU). A global optimization considering all the previous states is adopted to maintain high localization accuracy and global consistency, yet its complexity of being linear to the number of all previous camera/IMU observations seriously impedes real-time implementation. We show that the global optimization can be solved efficiently at the complexity linear to the number of keyframes, and further realize a real-time dense 3D reconstruction system given the estimated camera states. Meanwhile, for the sake of robustness, we propose a novel loop-validity detector based on the estimated bias of the IMU state. By checking the consistency of camera movements, a false loop closure constraint introduces manifest inconsistency between the camera movements and IMU measurements. Experiments reveal that iDFusion owns superior reconstruction performance running in 25 fps on CPU computing of portable devices, under challenging yet practical scenarios including texture-less, motion blur, and repetitive contents.



---

##### 3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans

###### \[[pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_3D-SIS_3D_Semantic_Instance_Segmentation_of_RGB-D_Scans_CVPR_2019_paper.pdf)\] CVPR 2019

> We introduce 3D-SIS, a novel neural network architecture for 3D semantic instance segmentation in commodity RGB-D scans. The core idea of our method is to jointly learn from both geometric and color signal, thus enabling accurate instance predictions. Rather than operate solely on 2D frames, we observe that most computer vision applications have multi-view RGB-D input available, which we leverage to construct an approach for 3D instance segmentation that effectively fuses together these multi-modal inputs. Our network leverages high-resolution RGB input by associating 2D images with the volumetric grid based on the pose alignment of the 3D reconstruction. For each image, we first extract 2D features for each pixel with a series of 2D convolutions; we then backproject the resulting feature vector to the associated voxel in the 3D grid. This combination of 2D and 3D feature learning allows significantly higher accuracy object detection and instance segmentation than state-of-the-art alternatives. We show results on both synthetic and real-world public benchmarks, achieving an improvement in mAP of over 13 on real-world data.

介绍了一种用于商品RGB-D扫描中的3D语义实例分割的神经网络结构3D-SIS。该方法的核心思想是联合学习几何信号和颜色信号，从而实现精确的实例预测。我们观察到，大多数计算机视觉应用程序都有多视图RGB-D输入，而不是仅在2D帧上操作，我们利用它构建一种3D实例分割方法，有效地将这些多模式输入融合在一起。我们的网络利用高分辨率的RGB输入，通过将2D图像与基于三维重建的姿势对齐的三维网格相关联。对于每个图像，我们首先通过一系列二维卷积提取每个像素的二维特征；然后将生成的特征向量反向投影到三维网格中的相关体素。这种二维和三维特征学习的结合使得目标检测和实例分割的精确度大大高于最先进的方法。我们展示了综合和真实世界公共基准的结果，在真实世界数据的地图上实现了13个以上的改进



----

##### BAD SLAM: Bundle Adjusted Direct RGB-D SLAM

###### \[[pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.pdf)]  \[[project page](www.eth3d.net)] CVPR 2019

> A key component of Simultaneous Localization and Mapping (SLAM) systems is the joint optimization of the estimated 3D map and camera trajectory. Bundle adjustment (BA) is the gold standard for this. Due to the large number of variables in dense RGB-D SLAM, previous work has focused on approximating BA. In contrast, in this paper we present a novel, fast direct BA formulation which we implement in a real-time dense RGB-D SLAM algorithm. In addition, we show that direct RGB-D SLAM systems are highly sensitive to rolling shutter, RGB and depth sensor synchronization, and calibration errors. In order to facilitate state-of-the-art research on direct RGB-D SLAM, we propose a novel, well-calibrated benchmark for this task that uses synchronized global shutter RGB and depth cameras. It includes a training set, a test set without public ground truth, and an online evaluation service. We observe that the ranking of methods changes on this dataset compared to existing ones, and our proposed algorithm outperforms all other evaluated SLAM methods. Our benchmark and our open source SLAM algorithm are available at: www.eth3d.net

同步定位与建图（SLAM）系统的一个重要组成部分是估计的三维地图和摄像机轨迹的联合优化。捆绑调整（BA）是这方面的金标准。由于稠密RGB-D SLAM中变量较多，以往的工作主要集中在BA的逼近上。相比之下，本文提出了一种新的、快速的直接BA公式，并在实时密集RGB-D SLAM算法中实现。此外，我们还发现直接RGB-D SLAM系统对滚动快门、RGB和深度传感器同步以及校准误差高度敏感。为了促进对直接RGB-D SLAM的最新研究，我们提出了一个新的、经过良好校准的基准，它使用同步的全局快门RGB和深度相机。它包括一个训练集，一个没有公开真相的测试集，以及一个在线评估服务。我们观察到该数据集上的方法排序与现有方法相比发生了变化，并且我们提出的算法优于所有其他评估的SLAM方法。我们的基准测试和开源SLAM算法可从以下网址获得：www.eth3d.net



----

##### Near-Eye Display and Tracking Technologies for Virtual and Augmented Reality

###### \[[pdf](https://dro.dur.ac.uk/27854/1/27854.pdf)]  Computer Graphics Forum 2019

> Virtual and augmented reality (VR/AR) are expected to revolutionise entertainment, healthcare, communication and the manufacturing industries among many others. Near-eye displays are an enabling vessel for VR/AR applications, which have to tackle many challenges related to ergonomics, comfort, visual quality and natural interaction. These challenges are related to the core elements of these near-eye display hardware and tracking technologies. In this state-of-the-art report, we investigate the background theory of perception and vision as well as the latest advancements in display engineering and tracking technologies. We begin our discussion by describing the basics of light and image formation. Later, we recount principles of visual perception by relating to the human visual system. We provide two structured overviews on state-of-the-art near-eye display and tracking technologies involved in such near-eye displays. We conclude by outlining unresolved research questions to inspire the next generation of researchers.

虚拟和增强现实（VR/AR）预计将彻底改变娱乐、医疗、通信和制造业等许多行业。近眼显示是虚拟现实/增强现实应用的一个有利载体，它必须解决许多与人体工程学、舒适性、视觉质量和自然交互有关的挑战。这些挑战与这些近眼显示硬件和跟踪技术的核心元素有关。在这篇最新的报告中，我们将研究感知和视觉的背景理论以及显示工程和跟踪技术的最新进展。我们从描述光和图像形成的基本原理开始讨论。之后，我们通过与人类视觉系统的联系来叙述视觉感知的原理。我们提供了两个结构化的概述，关于最先进的近眼显示和跟踪技术涉及到这种近眼显示。最后，我们概述了尚未解决的研究问题，以激励下一代研究人员



----

##### Multi-Robot Collaborative Dense Scene Reconstruction

###### \[[pdf](https://dl.acm.org/doi/pdf/10.1145/3306346.3322942)] TOG 2019

> We present an autonomous scanning approach which allows multiple robots to perform collaborative scanning for dense 3D reconstruction of unknown indoor scenes. Our method plans scanning paths for several robots, allowing them to efficiently coordinate with each other such that the collective scanning coverage and reconstruction quality is maximized while the overall scanning effort is minimized. To this end, we define the problem as a dynamic task assignment and introduce a novel formulation based on Optimal Mass Transport (OMT). Given the currently scanned scene, a set of task views are extracted to cover scene regions which are either unknown or uncertain. These task views are assigned to the robots based on the OMT optimization. We then compute for each robot a smooth path over its assigned tasks by solving an approximate traveling salesman problem. In order to showcase our algorithm, we implement a multi-robot auto-scanning system. Since our method is computationally efficient, we can easily run it in real time on commodity hardware, and combine it with online RGB-D reconstruction approaches. In our results, we show several real-world examples of large indoor environments; in addition, we build a benchmark with a series of carefully designed metrics for quantitatively evaluating multi-robot autoscanning. Overall, we are able to demonstrate high-quality scanning results with respect to reconstruction quality and scanning efficiency, which significantly outperforms existing multi-robot exploration systems.

我们提出了一种自主扫描方法，允许多个机器人协同扫描，对未知的室内场景进行密集的三维重建。我们的方法为多个机器人规划扫描路径，使它们能够有效地相互协调，从而在最大程度上提高扫描覆盖率和重建质量，同时使整体扫描工作量最小化。为此，我们将该问题定义为一个动态任务分配问题，并引入了一种新的基于最优质量传输（OMT）的计算公式。针对当前扫描的场景，提取一组任务视图来覆盖未知或不确定的场景区域。这些任务视图是基于OMT优化分配给机器人的。然后我们通过求解一个近似的旅行商问题来计算每个机器人在其指定任务上的平滑路径。为了展示我们的算法，我们实现了一个多机器人自动扫描系统。由于我们的方法计算效率高，我们可以很容易地在商品硬件上实时运行，并与在线RGB-D重建方法相结合。在我们的结果中，我们展示了几个真实的大型室内环境的例子；此外，我们建立了一个基准与一系列精心设计的指标，以定量评估多机器人自动扫描。总体而言，我们能够展示高质量的扫描结果，在重建质量和扫描效率方面，这明显优于现有的多机器人探测系统。



---

##### TextureFusion: High-Quality Texture Acquisition for Real-Time RGB-D Scanning

###### \[[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_TextureFusion_High-Quality_Texture_Acquisition_for_Real-Time_RGB-D_Scanning_CVPR_2020_paper.pdf)] [[code]((https://github.com/KAIST-VCLAB/texturefusion.git))]  CVPR 2020

> Real-time RGB-D scanning technique has become widely used to progressively scan objects with a hand-held sensor. Existing online methods restore color information per voxel, and thus their quality is often limited by the tradeoff between spatial resolution and time performance. Also, such methods often suffer from blurred artifacts in the captured texture. Traditional offline texture mapping methods with non-rigid warping assume that the reconstructed geometry and all input views are obtained in advance, and the optimization takes a long time to compute mesh parameterization and warp parameters, which prevents them from being used in real-time applications. In this work, we propose a progressive texture-fusion method specially designed for real-time RGB-D scanning. To this end, we first devise a novel texture-tile voxel grid, where texture tiles are embedded in the voxel grid of the signed distance function, allowing for high-resolution texture mapping on the low-resolution geometry volume. Instead of using expensive mesh parameterization, we associate vertices of implicit geometry directly with texture coordinates. Second, we introduce real-time texture warping that applies a spatiallyvarying perspective mapping to input images so that texture warping efficiently mitigates the mismatch between the intermediate geometry and the current input view. It allows us to enhance the quality of texture over time while updating the geometry in real-time. The results demonstrate that the quality of our real-time texture mapping is highly competitive to that of exhaustive offline texture warping methods. Our method is also capable of being integrated into existing RGB-D scanning frameworks



---

##### Geometry-Aware ICP for Scene Reconstruction from RGB-D Camera

###### [[pdf](http://www.sci-hub.ren/10.1007/s11390-019-1928-6)] JOURNAL OF COMPUTER SCIENCE AND TECHNOLOGY 2019

> The Iterative Closest Point (ICP) scheme has been widely used for the registration of surfaces and point clouds. However, when working on depth image sequences where there are large geometric planes with small (or even without) details, existing ICP algorithms are prone to tangential drifting and erroneous rotational estimations due to input device errors. In this paper, we propose a novel ICP algorithm that aims to overcome such drawbacks, and provides significantly stabler registration estimation for simultaneous localization and mapping (SLAM) tasks on RGB-D camera inputs. In our approach, the tangential drifting and the rotational estimation error are reduced by: 1) updating the conventional Euclidean distance term with the local geometry information, and 2) introducing a new camera stabilization term that prevents improper camera movement in the calculation. Our approach is simple, fast, effective, and is readily integratable with previous ICP algorithms. We test our new method with the TUM RGB-D SLAM dataset on state-of-the-art real-time 3D dense reconstruction platforms, i.e., ElasticFusion and Kintinuous. Experiments show that our new strategy outperforms all previous ones on various RGB-D data sequences under different combinations of registration systems and solutions.



---

##### Joint Texture and Geometry Optimization for RGB-D Reconstruction

###### [[pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_Joint_Texture_and_Geometry_Optimization_for_RGB-D_Reconstruction_CVPR_2020_paper.pdf)] CVPR 2020

> Due to inevitable noises and quantization error, the reconstructed 3D models via RGB-D sensors always accompany geometric error and camera drifting, which consequently lead to blurring and unnatural texture mapping results. Most of the 3D reconstruction methods focus on either geometry refinement or texture improvement respectively, which subjectively decouples the inter-relationship between geometry and texture. In this paper, we propose a novel approach that can jointly optimize the camera poses, texture and geometry of the reconstructed model, and color consistency between the key-frames. Instead of computing ShapeFrom-Shading (SFS) expensively, our method directly optimizes the reconstructed mesh according to color and geometric consistency and high-boost normal cues, which can effectively overcome the texture-copy problem generated by SFS and achieve more detailed shape reconstruction. As the joint optimization involves multiple correlated terms, therefore, we further introduce an iterative framework to interleave the optimal state. The experiments demonstrate that our method can recover not only fine-scale geometry but also high-fidelity texture.



---

##### Noise-Resilient Reconstruction of Panoramas and 3D Scenes using Robot-Mounted Unsynchronized Commodity RGB-D Cameras

###### [[pdf](http://orca.cf.ac.uk/130657/1/DepthPano-TOG2020.pdf)] TOG2020

>We present a two-stage approach to irst constructing 3D panoramas and then stitching them for noise-resilient reconstruction of large-scale indoor scenes. Our approach requires multiple unsynchronized RGB-D cameras, mounted on a robot platform which can perform in-place rotations at diferent locations in a scene. Such cameras rotate on a common (but unknown) axis, which provides a novel perspective for coping with unsynchronized cameras, without requiring suicient overlap of their Field-of-View (FoV). Based on this key observation, we propose novel algorithms to track these cameras simultaneously. Furthermore, during the integration of raw frames onto an equirectangular panorama, we derive uncertainty estimates from multiple measurements assigned to the same pixels. This enables us to appropriately model the sensing noise and consider its inluence, so as to achieve better noise resilience, and improve the geometric quality of each panorama and the accuracy of global inter-panorama registration. We evaluate and demonstrate the performance of our proposed method for enhancing the geometric quality of scene reconstruction from both real-world and synthetic scans.


##### DeepDeform: Learning Non-Rigid RGB-D Reconstruction With Semi-Supervised Data

[[pdf](https://arxiv.org/pdf/1912.04302.pdf)] [[data](https://github.com/AljazBozic/DeepDeform)]

>Applying data-driven approaches to non-rigid 3D reconstruction has been difficult, which we believe can be attributed to the lack of a large-scale training corpus. Unfortunately, this method fails for important cases such as highly non-rigid deformations. We first address this problem of lack of data by introducing a novel semi-supervised strategy to obtain dense inter-frame correspondences from a sparse set of annotations. This way, we obtain a large dataset of 400 scenes, over 390,000 RGB-D frames, and 5,533 densely aligned frame pairs; in addition, we provide a test set along with several metrics for evaluation. Based on this corpus, we introduce a data-driven nonrigid feature matching approach, which we integrate into an optimization-based reconstruction pipeline. Here, we propose a new neural network that operates on RGB-D frames, while maintaining robustness under large non-rigid deformations and producing accurate predictions. Our approach significantly outperforms existing non-rigid reconstruction methods that do not use learned data terms, as well as learning-based approaches that only use self-supervision.

>将数据驱动的方法应用于非刚性三维重建一直是一个困难的问题，我们认为这可以归因于缺乏大规模的训练语料库。不幸的是，这种方法对于高度非刚性变形等重要情况是失败的。我们首先通过引入一种新的半监督策略来解决数据不足的问题，从稀疏的注释集中获得密集的帧间对应。通过这种方式，我们获得了一个包含400个场景、超过390000个RGB-D帧和5533个密集对齐帧对的大型数据集；此外，我们还提供了一个测试集以及一些用于评估的度量。基于此语料库，我们引入了一种数据驱动的非刚性特征匹配方法，并将其集成到基于优化的重建流水线中。在这里，我们提出了一种新的神经网络，它可以在RGB-D框架上工作，同时在大的非刚性变形下保持鲁棒性并产生精确的预测。我们的方法明显优于现有的不使用学习数据项的非刚性重建方法，以及只使用自我监督的基于学习的方法。

---
##### Visual Camera Re-Localization from RGB and RGB-D Images Using DSAC

[[pdf](https://arxiv.org/pdf/2002.12324.pdf)]

>We describe a learning-based system that estimates the camera position and orientation from a single input image relative to a known environment. The system is flexible w.r.t. the amount of information available at test and at training time, catering to different applications. Input images can be RGB-D or RGB, and a 3D model of the environment can be utilized for training but is not necessary. In the minimal case, our system requires only RGB images and ground truth poses at training time, and it requires only a single RGB image at test time. The framework consists of a deep neural network and fully differentiable pose optimization. The neural network predicts so called scene coordinates, i.e. dense correspondences between the input image and 3D scene space of the environment. The pose optimization implements robust fitting of pose parameters using differentiable RANSAC (DSAC) to facilitate end-to-end training. The system, an extension of DSAC++ and referred to as DSAC*, achieves state-of-the-art accuracy an various public datasets for RGB-based re-localization, and competitive accuracy for RGB-D based re-localization.

>我们描述了一个基于学习的系统，它可以根据单个输入图像相对于已知环境估计相机的位置和方向。该系统是灵活的w.r.t.测试和培训时可用的信息量，以满足不同的应用。输入图像可以是RGB-D或RGB，环境的三维模型可以用于训练，但不是必需的。在最小的情况下，我们的系统在训练时只需要RGB图像和地面真实姿态，并且在测试时只需要一个RGB图像。该框架由深度神经网络和全微分位姿优化组成。神经网络预测所谓的场景坐标，即输入图像与环境的三维场景空间之间的密集对应。位姿优化利用可微RANSAC（DSAC）实现了姿态参数的鲁棒拟合，方便了端到端的训练。该系统是DSAC++的扩展，被称为DSAC*，它在基于RGB的重定位中实现了最先进的精度和各种公共数据集，以及基于RGB-D的重定位的竞争精度。

##### SPSG: Self-Supervised Photometric Scene Generation from RGB-D Scans
[[pdf](https://arxiv.org/pdf/2006.14660.pdf)]

>We present SPSG, a novel approach to generate high-quality, colored 3D models of scenes from RGB-D scan observations by learning to infer unobserved scene geometry and color in a self-supervised fashion. Our self-supervised approach learns to jointly inpaint geometry and color by correlating an incomplete RGB-D scan with a more complete version of that scan. Notably, rather than relying on 3D reconstruction losses to inform our 3D geometry and color reconstruction, we propose adversarial and perceptual losses operating on 2D renderings in order to achieve high-resolution, high-quality colored reconstructions of scenes. This exploits the high-resolution, self-consistent signal from individual raw RGB-D frames, in contrast to fused 3D reconstructions of the frames which exhibit inconsistencies from view-dependent effects, such as color balancing or pose inconsistencies. Thus, by informing our 3D scene generation directly through 2D signal, we produce high-quality colored reconstructions of 3D scenes, outperforming state of the art on both synthetic and real data.

>我们提出了一种新的SPSG方法，通过学习以一种自监督的方式来推断未观测到的场景几何和颜色，从而从RGB-D扫描观测数据中生成高质量的彩色3D场景模型。我们的自我监督方法通过将不完整的RGB-D扫描与该扫描的更完整版本相关联来学习联合修复几何体和颜色。值得注意的是，我们不依赖3D重建损失来告知我们的3D几何和颜色重建，我们建议在2D渲染上操作对抗性和感知损失，以实现场景的高分辨率、高质量的彩色重建。这利用了来自单个原始RGB-D帧的高分辨率、自洽信号，而帧的融合3D重建则显示出与视点相关的效果不一致，例如颜色平衡或姿势不一致。因此，通过直接通过2D信号通知我们的3D场景生成，我们可以生成高质量的3D场景彩色重建，在合成数据和真实数据方面都优于现有技术。

---

##### X-Section: Cross-Section Prediction for Enhanced RGB-D Fusion

[[pdf](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nicastro_X-Section_Cross-Section_Prediction_for_Enhanced_RGB-D_Fusion_ICCV_2019_paper.pdf)]

>Detailed 3D reconstruction is an important challenge with application to robotics, augmented and virtual reality, which has seen impressive progress throughout the past years. Advancements were driven by the availability of depth cameras (RGB-D), as well as increased compute power, e.g. in the form of GPUs – but also thanks to inclusion of machine learning in the process. Here, we propose X-Section, an RGB-D 3D reconstruction approach that leverages deep learning to make object-level predictions about thicknesses that can be readily integrated into a volumetric multi-view fusion process, where we propose an extension to the popular KinectFusion approach. In essence, our method allows to complete shapes in general indoor scenes behind what is sensed by the RGB-D camera, which may be crucial e.g. for robotic manipulation tasks or efficient scene exploration. Predicting object thicknesses rather than volumes allows us to work with comparably high spatial resolution without exploding memory and training data requirements on the employed Convolutional Neural Networks. In a series of qualitative and quantitative evaluations, we demonstrate how we accurately predict object thickness and reconstruct general 3D scenes containing multiple objects.

>详细的三维重建是机器人学、增强现实和虚拟现实应用中的一个重要挑战，在过去的几年里取得了令人瞩目的进展。这些进步是由深度摄影机（RGB-D）的可用性以及计算能力的提高（例如以GPU的形式）推动的，但也要归功于过程中包含了机器学习。在这里，我们提出了X-Section，一种RGB-D三维重建方法，它利用深度学习来对厚度进行对象级预测，可以很容易地集成到体积多视图融合过程中，在这里我们提出了对流行的KinectFusion方法的扩展。本质上，我们的方法允许在一般室内场景中完成由RGB-D摄像机感知到的形状，这对于机器人操作任务或有效的场景探索可能是至关重要的。预测物体的厚度而不是体积，使我们能够在不消耗内存和训练所用卷积神经网络的数据需求的情况下，以相当高的空间分辨率工作。在一系列的定性和定量评估中，我们展示了如何准确地预测物体厚度和重建包含多个物体的一般三维场景。

---

##### A Vertex-to-Edge Weighted Closed-Form Method for Dense RGB-D Indoor SLAM

[[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8649628)]

>RGB-D dense mapping systems are widely used for indoor navigation and augmented reality. However, the drift in egomotion estimation systems accumulates as the systems expand over space and time. In this paper, we develop a vertex-to-edge weighted closed-form algorithm to reduce camera drift for dense RGB-D indoor simultaneous localization and mapping (SLAM). First, due to the high-frame rate of common RGB-D sensors, in our approach, we employ a robust key-frame selection strategy. We then consider the typical depth noise properties of the sensor and exploit both visual and geometric information to determine the camera motion of the key frames. To enforce global consistency, we employ a combination of techniques to enhance the efficiency and accuracy of loop closure detection and then factor graph optimization is used to mitigate trajectory drift; specifically, the graph edges are weighted by residual errors. To demonstrate its accuracy and robustness, our SLAM system is tested on seven sets of sequences from publicly available benchmark datasets collected with Kinect v1 and on three sets of sequences taken with a handheld structure sensor. In a direct comparison, our approach yields significantly less trajectory error than several other stateof-the-art methods.

>RGB-D密集映射系统广泛应用于室内导航和增强现实。然而，自我运动估计系统中的漂移随着系统在空间和时间上的扩展而累积。针对密集RGB-D室内同步定位与映射（SLAM），提出了一种点到边加权闭式算法来减小摄像机漂移。首先，由于普通RGB-D传感器的高帧速率，在我们的方法中，我们采用了一种鲁棒的关键帧选择策略。然后考虑传感器的典型深度噪声特性，利用视觉和几何信息确定关键帧的摄像机运动。为了保证全局一致性，我们采用多种技术来提高环路闭合检测的效率和精度，然后利用因子图优化来减小轨迹漂移；具体地说，通过残差加权图边缘。为了证明它的准确性和鲁棒性，我们的SLAM系统在使用Kinect v1收集的公共可用基准数据集的七组序列和手持结构传感器采集的三组序列上进行了测试。与其他几种最先进的方法相比，我们的方法产生的轨迹误差要小得多。