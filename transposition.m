clear; clc;
format long g;

% =========================================================
% transposition.m
% 功能：从 MATLAB stereoParameters 对象中无损导出 OpenCV/Python 使用的双目标定参数。
%
% 重要说明：
% 1. Python 后续只读取 outputMat，不读取 txt。
% 2. .mat 中保存的是 MATLAB double，二进制保存，不会按 14/15 位字符串截断。
% 3. outputTxt 只用于人工核对，不作为 Python 参数来源。
% =========================================================

% ================= 配置路径 =================
inputFile = 'F:/Camera/matlab.mat';
outputMat = 'F:/Camera/stereo_params_plain.mat';
outputTxt = 'F:/Camera/stereo_params_matlab_side.txt';

% ================= 1. 加载 stereoParameters =================
if ~exist(inputFile, 'file')
    error('找不到输入文件: %s', inputFile);
end

S = load(inputFile);
names = fieldnames(S);

stereoParams = [];
for i = 1:numel(names)
    val = S.(names{i});
    if isa(val, 'stereoParameters')
        stereoParams = val;
        break;
    end
end

if isempty(stereoParams)
    error('未在 mat 文件中找到 stereoParameters 对象');
end

% ================= 2. 提取并转换为 OpenCV 约定 =================
% MATLAB 的 IntrinsicMatrix 存储形式与 OpenCV K 矩阵互为转置。
K1_matlab_raw = double(stereoParams.CameraParameters1.IntrinsicMatrix);
K2_matlab_raw = double(stereoParams.CameraParameters2.IntrinsicMatrix);
K1 = K1_matlab_raw';
K2 = K2_matlab_raw';

% 畸变顺序统一为 OpenCV: [k1, k2, p1, p2, k3]
d1_rad = double(stereoParams.CameraParameters1.RadialDistortion);
d2_rad = double(stereoParams.CameraParameters2.RadialDistortion);
d1_tan = double(stereoParams.CameraParameters1.TangentialDistortion);
d2_tan = double(stereoParams.CameraParameters2.TangentialDistortion);

if isempty(d1_tan), d1_tan = [0 0]; end
if isempty(d2_tan), d2_tan = [0 0]; end
if numel(d1_rad) < 3, d1_rad(3) = 0; end
if numel(d2_rad) < 3, d2_rad(3) = 0; end

dist1 = double([d1_rad(1), d1_rad(2), d1_tan(1), d1_tan(2), d1_rad(3)]);
dist2 = double([d2_rad(1), d2_rad(2), d2_tan(1), d2_tan(2), d2_rad(3)]);

% MATLAB stereoParameters 中 Camera2 相对 Camera1 的外参。
% 为了匹配 OpenCV stereoRectify，这里使用原先你代码中的转换：R = RotationOfCamera2'。
R_matlab_raw = double(stereoParams.RotationOfCamera2);
T_matlab_raw = double(stereoParams.TranslationOfCamera2);   % 1x3 行向量，通常单位是 WorldUnits，例如 mm
R = R_matlab_raw';
T = T_matlab_raw';                                          % 3x1 列向量

% 图像尺寸统一为 OpenCV 顺序: [width, height]
imgSize = [double(stereoParams.CameraParameters1.ImageSize(2)), ...
           double(stereoParams.CameraParameters1.ImageSize(1))];

% 保存一些元信息，方便核对
try
    worldUnits = stereoParams.WorldUnits;
catch
    worldUnits = 'unknown';
end

try
    meanReprojectionError = double(stereoParams.MeanReprojectionError);
catch
    meanReprojectionError = NaN;
end

% ================= 3. 二进制无损保存 =================
% 注意：这个 .mat 是 Python 唯一参数来源。不要从 txt 复制参数到 Python。
save(outputMat, ...
    'K1', 'K2', 'dist1', 'dist2', 'R', 'T', 'imgSize', ...
    'K1_matlab_raw', 'K2_matlab_raw', ...
    'd1_rad', 'd2_rad', 'd1_tan', 'd2_tan', ...
    'R_matlab_raw', 'T_matlab_raw', ...
    'worldUnits', 'meanReprojectionError', ...
    '-v7');

% ================= 4. 文本核对文件，仅供人工查看 =================
% 使用 %.17g 是为了尽可能 round-trip 显示 double；但正式程序不要读取 txt。
fid = fopen(outputTxt, 'w');
if fid < 0
    error('无法创建文本文件: %s', outputTxt);
end

fprintf(fid, 'This txt file is only for human checking. Python must read stereo_params_plain.mat.\n');
fprintf(fid, 'worldUnits: %s\n', string(worldUnits));
fprintf(fid, 'meanReprojectionError: %.17g\n\n', meanReprojectionError);

writeMatrix(fid, 'K1', K1);
writeMatrix(fid, 'K2', K2);
writeVector(fid, 'dist1 [k1 k2 p1 p2 k3]', dist1);
writeVector(fid, 'dist2 [k1 k2 p1 p2 k3]', dist2);
writeMatrix(fid, 'R', R);
writeVector(fid, 'T', T);
writeVector(fid, 'imgSize [width height]', imgSize);

fclose(fid);

fprintf('MATLAB 导出完成。\n');
fprintf('二进制参数文件: %s\n', outputMat);
fprintf('人工核对文本:   %s\n', outputTxt);
fprintf('Baseline norm: %.17g %s\n', norm(T), string(worldUnits));

% ================= 局部函数 =================
function writeMatrix(fid, name, M)
    fprintf(fid, '--- %s ---\n', name);
    for r = 1:size(M, 1)
        for c = 1:size(M, 2)
            fprintf(fid, '%.17g ', M(r, c));
        end
        fprintf(fid, '\n');
    end
    fprintf(fid, '\n');
end

function writeVector(fid, name, v)
    fprintf(fid, '--- %s ---\n', name);
    v = v(:);
    for i = 1:numel(v)
        fprintf(fid, '%.17g\n', v(i));
    end
    fprintf(fid, '\n');
end
