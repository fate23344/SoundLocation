function tdoa_2d_localization_square_array()
    % 基于TDOA的2D声源定位系统（正方形麦克风阵列）
    close all;
    clear;
    clc;
    
    % 参数设置
    c = 343;                % 声速(m/s)
    L = 0.0457;            % 正方形边长(m)
    fs = 48000;            % 采样频率(Hz) 修改为48kHz

    % 麦克风位置(正方形阵列，中心在坐标原点)
    mic_pos = [ L/2,  L/2;   % 右上
               -L/2,  L/2;   % 左上
               -L/2, -L/2;   % 左下
                L/2, -L/2];  % 右下
    
    % 读取四个麦克风的音频文件
    [y1, fs1] = audioread('mic-02 .wav');
    [y2, fs2] = audioread('mic-03 .wav');
    [y3, fs3] = audioread('mic-04 .wav');
    [y4, fs4] = audioread('mic-05 .wav');
    
    % 确保采样率一致且为48kHz
    if ~isequal([fs1,fs2,fs3,fs4], 48000*ones(1,4))
        error('采样率不一致或未达48kHz要求！');
    end
    
    % 统一信号长度
    min_len = min([length(y1), length(y2), length(y3), length(y4)]);
    y1 = y1(1:min_len);
    y2 = y2(1:min_len);
    y3 = y3(1:min_len);
    y4 = y4(1:min_len);
    
    % 带通滤波器设计（300Hz-3.5kHz）
    [b, a] = butter(4, [300, 3500]/(fs/2), 'bandpass');
    y1_filt = filtfilt(b, a, y1);
    y2_filt = filtfilt(b, a, y2);
    y3_filt = filtfilt(b, a, y3);
    y4_filt = filtfilt(b, a, y4);
    
    % 计算互相关函数（以第一个麦克风为参考）
    [corr12, lag12] = xcorr(y2_filt, y1_filt);
    [corr13, lag13] = xcorr(y3_filt, y1_filt);
    [corr14, lag14] = xcorr(y4_filt, y1_filt);
    
    % 改进峰值检测：添加抛物线插值
    [tau12, sample_shift12] = parabolic_interpolation(corr12, lag12, fs);
    [tau13, sample_shift13] = parabolic_interpolation(corr13, lag13, fs);
    [tau14, sample_shift14] = parabolic_interpolation(corr14, lag14, fs);
    
    fprintf('改进后TDOA测量：\n');
    fprintf('TDOA12: %.6f s (%.2f samples)\n', tau12, sample_shift12);
    fprintf('TDOA13: %.6f s (%.2f samples)\n', tau13, sample_shift13);
    fprintf('TDOA14: %.6f s (%.2f samples)\n', tau14, sample_shift14);
    
    % 优化求解参数设置
    options = optimoptions('lsqnonlin', ...
        'Display', 'iter', ...
        'Algorithm', 'levenberg-marquardt', ...
        'FunctionTolerance', 1e-6);
    
    % 初始猜测（阵列前方）
    x0 = [0, 0.5];  % [x, y]（假设在阵列正前方）
    array_center = mean(mic_pos);  % 阵列几何中心
    
    % 优化求解
    est_pos = lsqnonlin(@(x) tdoa_error_square(x, mic_pos, [0; tau12; tau13; tau14], c), ...
                        x0, [], [], options);
    
    % 结果分析与显示
    fprintf('\n估计声源位置: (%.3f m, %.3f m)\n', est_pos(1), est_pos(2));
    
    % 方位角计算（相对于阵列中心）
    rel_vector = est_pos - array_center;
    azimuth = atan2d(rel_vector(2), rel_vector(1));
    fprintf('方位角: %.1f° (0°指向正x方向，90°指向正y方向)\n', azimuth);
    
    % 可视化结果增强
    visualize_results(mic_pos, est_pos, array_center, azimuth, ...
                      {corr12, lag12}, {corr13, lag13}, {corr14, lag14}, fs);
end

function err = tdoa_error_square(x, mic_pos, measured_tdoa, c)
    % 2D定位误差计算（适用于正方形阵列）
    ref_dist = norm(x - mic_pos(1,:));
    pred_tdoa = zeros(4,1);
    
    for i = 1:4
        pred_tdoa(i) = (norm(x - mic_pos(i,:)) - ref_dist)/c;
    end
    
    err = pred_tdoa(2:4) - measured_tdoa(2:4);
end

function [tau, sample_shift] = parabolic_interpolation(corr, lag, fs)
    % 抛物线插值精化峰值检测
    [~, idx] = max(abs(corr));
    if idx == 1 || idx == length(corr)
        sample_shift = lag(idx);
        tau = sample_shift / fs;
        return;
    end
    
    % 抛物线模型系数求解
    y0 = abs(corr(idx-1));
    y1 = abs(corr(idx));
    y2 = abs(corr(idx+1));
    
    delta = (y2 - y0) / (2*(y0 - 2*y1 + y2));
    sample_shift = lag(idx) + delta;
    tau = sample_shift / fs;
end

function visualize_results(mic_pos, est_pos, center, azimuth, corr12, corr13, corr14, fs)
    % 增强可视化函数（修正极坐标问题）
    figure('Name', '阵列配置与定位结果', 'Position', [100 100 800 800]);
    scatter(mic_pos(:,1), mic_pos(:,2), 120, 'b', 'filled', 'MarkerEdgeColor', 'k');
    hold on;
    plot(est_pos(1), est_pos(2), 'r*', 'MarkerSize', 15, 'LineWidth', 2);
    plot(center(1), center(2), 'k+', 'MarkerSize', 15, 'LineWidth', 2);
    
    % 绘制阵列边缘
    plot([mic_pos(1:end,1); mic_pos(1,1)], [mic_pos(1:end,2); mic_pos(1,2)],...
         'b--', 'LineWidth', 1.5);
    
    % 修改后的方位角显示（使用笛卡尔坐标系）
    [x_end, y_end] = pol2cart(deg2rad(azimuth), 0.15); % 0.15m长度指示线
    quiver(center(1), center(2), x_end, y_end, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 1);
    
    % 标注与样式设置
    text(est_pos(1)+0.02, est_pos(2), sprintf('(%.3f, %.3f)m', est_pos(1), est_pos(2)),...
        'FontSize', 10, 'Color', 'r');
    title(sprintf('声源定位结果 - 方位角%.1f°', azimuth));
    xlabel('X (m)'); ylabel('Y (m)');
    legend('麦克风', '声源位置', '阵列中心', '阵列轮廓', '方位指示',...
           'Location', 'best');
    grid on; axis equal;
    
    % 互相关结果可视化优化
    figure('Name', '互相关函数分析', 'Position', [900 100 800 600]);
    subplot(3,1,1);
    plot_correlation(corr12{2}/fs, corr12{1}, 'Mic1-Mic2', fs);
    subplot(3,1,2);
    plot_correlation(corr13{2}/fs, corr13{1}, 'Mic1-Mic3', fs);
    subplot(3,1,3);
    plot_correlation(corr14{2}/fs, corr14{1}, 'Mic1-Mic4', fs);
end

function plot_correlation(time, corr, title_str, fs)
    % 互相关绘图标准化
    plot(time, corr, 'LineWidth', 1.2);
    title([title_str ' 互相关分析 (Fs=' num2str(fs/1e3) 'kHz)']);
    xlabel('时间差 (s)'); ylabel('相关系数');
    xlim([-0.002, 0.002]); % 聚焦关键时延区域
    grid on;
    ax = gca;
    ax.XAxisLocation = 'origin';
    ax.YAxisLocation = 'origin';
end
