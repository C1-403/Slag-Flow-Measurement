% --- MATLAB 转换脚本 (Python 兼容版) ---

% 1. 加载您原始的 .mat 文件
%    请将 'cameraParams_goucao3.mat' 替换成您的实际文件名
fprintf('正在加载原始相机参数文件 "cameraParams.mat"...\n');
load('cameraParams.mat');

% 检查 'cameraParams' 变量是否存在，以确保文件加载成功
if ~exist('cameraParams', 'var')
    error('错误: 在加载的 .mat 文件中未找到 "cameraParams" 变量。请检查文件名和文件内容。');
end

fprintf('成功加载 cameraParams 对象。\n');

% 2. 从 cameraParameters 对象中提取所有需要的数据
fprintf('正在从 cameraParameters 对象中提取核心数据...\n');

% 内参和畸变系数
intrinsicMatrix = cameraParams.IntrinsicMatrix;
radialDistortion = cameraParams.RadialDistortion;
tangentialDistortion = cameraParams.TangentialDistortion;

% 所有标定图像的外参（旋转矩阵和平移向量），用于后续三维计算
rotationMatrices = cameraParams.RotationMatrices;
translationVectors = cameraParams.TranslationVectors;

% 3. 将这些数据保存到一个新的 .mat 文件中
output_filename = 'camera_params_for_python.mat';
fprintf('正在将提取的数据保存到新文件 "%s"...\n', output_filename);

%    这个新文件将只包含纯粹的矩阵和向量，Python的scipy库可以轻松读取。
%    我们使用 '-v7' 格式来保证与 scipy.io.loadmat 的最佳兼容性。
save(output_filename, ...
    'intrinsicMatrix', ...
    'radialDistortion', ...
    'tangentialDistortion', ...
    'rotationMatrices', ...
    'translationVectors', ...
    '-v7');

% 4. 提示用户操作完成
fprintf('\n'); % 添加一个空行，让输出更美观
disp('转换完成！');
fprintf('新的 .mat 文件 "%s" 已经成功生成。\n', output_filename);
disp('请在您的 Python 代码中使用这个新文件。');

