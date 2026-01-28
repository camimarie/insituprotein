%% Loading data

% Add the Bio-Formats toolbox to your MATLAB path
addpath('bfmatlab/bfmatlab');

% Define the common part of the file names for your TIF images
commonFileName = '/Users/camillemitchell/MIT Dropbox/Camille Mitchell/temporary_analysis/02212025/CM_Fpep_';

% File name format: CM_experiment_condition_sample.tif eg. CM_Fpep_A1.tif

% Define sample info for different conditions - uncomment relevant samples
sampleInfo = {

    "A", "DMSO";
    "B", "DMSO:PITC";
    "C", "TFA";
    "D", "DMSO:PITC+TFA";
    "E", "DMSO:PITC+TFA+trypsin";

    % "F", "Phenylalanine (F)";
    % "A", "Alanine (A)";
    % "W", "Tryptophan (W)";
    % "Y", "Tyrosine (Y)";
    
    % "1mMHAtag", "1mM HA peptide",
    % "100uMHAtag", "100uM HA peptide",
    % "10uMHAtag", "10uM HA peptide",
    % "emptyHAtag", "empty"

    % "empty_5mM", "empty",
    % "Ala_5mM", "Ala 5mM",
    % "Trp_5mM", "Trp 5mM",
    % "Phe_5mM", "Phe 5mM",

    % "FAbF", "ClickP-F gel";
    % "FAbV", "ClickP-F gel";
    % "GAbF", "ClickP-G gel";
    % "GAbV", "ClickP-G gel";
    % "VAbF", "ClickP-V gel";
    % "VAbV", "ClickP-V gel";

};

numConditions = size(sampleInfo, 1);
numSamplesPerCondition = 3;
z_step = 10;
start_section = 0; % start of cross section
end_section = 430; % end of cross section
y_lim = 1000;
num_z = (end_section-start_section)/z_step; % number of z in the cross section considered

%% Cross-section plots from bulk fluorescence data - with TIFF FILES
% Define the desired x-axis range
desiredXRange = [0, end_section];

% Initialize arrays to store results
meanFluorescence = zeros(numConditions, numSamplesPerCondition, num_z); % start at 100 until 200 um, 10um steps
stdDevFluorescence = zeros(numConditions, num_z);
xAxisMicrometers = linspace(start_section, end_section, num_z); % Adjust range and number of slices
meanSampleFluorescence = zeros(numConditions, num_z);

for condition = 1:numConditions
    zStack = cell(1, numSamplesPerCondition);  % Initialize zStack for the current condition
    % Process each sample for the current condition
    for sample = 1:numSamplesPerCondition
        % Construct the file path for the current sample
        filePath = char(strcat(commonFileName, sampleInfo{condition, 1},num2str(sample),'.tif')); % num2str(sample) '_t' 
        disp(filePath)
        % Load TIFF images into zStack
        info = imfinfo(filePath);
        num_slices_sample = numel(info); % Get the number of slices in the TIFF stack
        zStack{sample} = cell(1, num_slices_sample); % Initialize cell array to store each slice
        
        % Load each slice of the z-stack
        for slice = 1:num_slices_sample
            zStack{sample}{slice} = imread(filePath, slice); % Load individual slice
        end
        
        % Process each element within zStack{sample}
        for i = 1:min(num_slices_sample, num_z)
            % Calculate mean fluorescence for the current element
            % Assuming your data is numeric (replace with actual data)
            currentElementData = zStack{sample}{i};
            meanFluorescence(condition, sample, i) = mean(currentElementData(:));
        end
    end
    
    % Compute the standard deviation for each slice across samples
    stdDevFluorescence(condition, :) = std(meanFluorescence(condition, :, :), 0, 2);
    % Compute the mean for each slice across samples
    meanSampleFluorescence(condition, :) = mean(meanFluorescence(condition, :, :), 2); % mean
end

% Define a set of color-blind-friendly colors
colors = [
    % For Edman conditions
    0.0 0.45 0.70; % Blue
    0.85 0.33 0.10; % Orange
    0.93 0.69 0.13; % Yellow
    0.49 0.18 0.56; % Purple
    % 0.47 0.67 0.19; % Green
    
    % For N-terminal conditions and Glyphic antibody read-outs
    % 0.30 0.75 0.93; % Cyan
    % 0.64 0.08 0.18  % Red
    % 0.70 0.70 0.70; % Neutral Gray
    %0.98 0.50 0.44; % Coral
];

% Initialize the figure
figure;
hold on;

% Process each condition
for condition = 1:numConditions
    color = colors(mod(condition-1, size(colors, 1)) + 1, :); % Cycle through colors
    meanCurve = squeeze(meanSampleFluorescence(condition, :));
    stdDevCurve = stdDevFluorescence(condition, :);
    
    % Plot the mean curve
    qw{condition} = plot(xAxisMicrometers, meanCurve, 'LineWidth', 2, 'Color', color);
    
    % Add shaded error bars for standard deviation
    fill([xAxisMicrometers, fliplr(xAxisMicrometers)], ...
         [meanCurve + stdDevCurve, fliplr(meanCurve - stdDevCurve)], ...
         color, 'EdgeColor', 'none', 'FaceAlpha', 0.3);
end

% Set axis limits
xlim(desiredXRange);
ylim([0 y_lim]); % adjust this accordingly

% Increase x-axis and y-axis tick label font size and remove the box around the figure
set(gca, 'FontSize', 20, 'LineWidth', 2, 'Box', 'off'); % Larger tick labels and thicker axes

% Add axis labels and title with larger font size
xlabel('Cross Section (µm)', 'FontSize', 24, 'FontWeight', 'bold'); % Larger x-axis label
ylabel('Fluorescence Intensity (a.u.)', 'FontSize', 24, 'FontWeight', 'bold'); % Larger y-axis label
title({['HA peptide staining with HA-antibody 647']}, 'FontSize', 28, 'FontWeight', 'bold'); % Larger title

% Create legend using a loop
legendHandles = cell(1, numConditions);
legendLabels = cell(1, numConditions);

for condition = 1:numConditions
    legendHandles{condition} = qw{condition};
    legendLabels{condition} = sampleInfo{condition, 2};
end

% Add legend and adjust its properties
legend(vertcat(legendHandles{:}), legendLabels{:}, 'Location', 'best', 'FontSize', 18, 'Box', 'off'); % Larger legend

%% Bar graphs max intensity projections - for TIFF files

% Update the start and end indices for the z stack
start_z = 1;
end_z = num_z;

% Initialize arrays to store the max stack and average intensity for each condition
maxStacks = zeros(1, numConditions);
maxStacksStd = zeros(1, numConditions);
averageIntensities = zeros(1, numConditions);
averageIntensitiesStd = zeros(1, numConditions);

for condition = 1:numConditions
    % Initialize variables to store mean intensities for each sample
    meanIntensities = zeros(end_z - start_z + 1, numSamplesPerCondition);
    
    % Initialize variables to store z-stacks for each sample
    maxIntensityZStacks = zeros(1, numSamplesPerCondition);
    averageIntensityZStacks = zeros(1, numSamplesPerCondition);
    
    % Process each sample for the current condition
    for sample = 1:numSamplesPerCondition
        % Construct the file path for the current sample
        filePath = char(strcat(commonFileName, sampleInfo{condition, 1}, num2str(sample),'.tif')); 
        disp(filePath)
        
        % Load TIFF images into zStack
        info = imfinfo(filePath);
        num_slices_sample = numel(info); % Get the number of slices in the TIFF stack
        zStack = cell(1, num_slices_sample); % Initialize cell array to store each slice
        
        % Load each slice of the z-stack
        for slice = 1:num_slices_sample
            zStack{slice} = imread(filePath, slice); % Load individual slice
        end
        
        % Process each element within the specified z stack range
        for z_stack = start_z:end_z
            % Calculate mean intensity for the current sample
            currentElementData = zStack{z_stack};
            meanIntensities(z_stack - start_z + 1, sample) = mean(currentElementData(:));
        end
        
        % Store the z-stack with the maximum mean intensity
        [~, maxMeanIndex] = max(meanIntensities(:, sample));
        maxIntensityZStacks(sample) = mean(mean(zStack{maxMeanIndex}));
        rawMaxIntensityData{condition}(sample) = maxIntensityZStacks(sample);

        % Get average intensity for each sample
        averageIntensityZStacks(sample) = mean(meanIntensities(:, sample));
        rawAverageIntensityData{condition}(sample) = averageIntensityZStacks(sample);

    end
    
    % Calculate max stack for the current condition
    maxStack = mean(maxIntensityZStacks);
    maxStacksStd(condition) = std(maxIntensityZStacks);
    maxStacks(condition) = maxStack;
    
    % Calculate average intensity for the current condition
    averageIntensity = mean(averageIntensityZStacks);
    averageIntensitiesStd(condition) = std(averageIntensityZStacks);
    averageIntensities(condition) = averageIntensity;
end

% Define a set of muted/pastel colors for the conditions
colors = [
    % For Edman conditions
    0.6 0.75 0.85; % Light Blue
    0.95 0.7 0.5;  % Soft Orange
    0.98 0.85 0.55; % Pale Yellow
    0.75 0.6 0.8;  % Lavender
    0.65 0.85 0.6; % Light Green

    % For the N-terminal peptide conditions
    % 0.30 0.75 0.93; % Cyan (original)
    % 0.67 0.85 0.90; % Soft Cyan
    % 
    % % 0.64 0.08 0.18; % Red (original)
    % 0.85 0.45 0.50; % Rosewood
    % 
    % % 0.70 0.70 0.70; % Neutral Gray (original)
    % 0.80 0.80 0.80; % Light Gray

    % 0.98 0.50 0.44; % Coral (original)
    %0.94 0.70 0.65; % Pale Coral
];

% Create a bar graph for average intensity
figure;

% Bar graph with individual colors for each condition
b = bar(averageIntensities, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);
hold on;

% Assign colors to each bar
for condition = 1:numel(averageIntensities)
    b.CData(condition, :) = colors(mod(condition-1, size(colors, 1)) + 1, :);
end

% Add error bars for standard deviation
x = (1:numel(sampleInfo(:, 2))).';
err = averageIntensitiesStd;
eb = errorbar(x, averageIntensities, err, 'k', 'linestyle', 'none');
eb.LineWidth = 1.5;

% Plot raw data points with jitter as black circles
jitterAmount = 0.1; % Adjust this value for the desired jitter spread
for condition = 1:numConditions
    jitteredX = condition + jitterAmount * (rand(1, numSamplesPerCondition) - 0.5);
    scatter(jitteredX, rawAverageIntensityData{condition}, 'o', ...
            'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'k');
end

% Customize axis appearance
set(gca, 'FontSize', 20, 'LineWidth', 1.5, 'Box', 'off'); % Larger ticks and remove box
xlabel('Conditions', 'FontSize', 24, 'FontWeight', 'bold');
ylabel('Fluorescence intensity (a.u.)', 'FontSize', 24, 'FontWeight', 'bold');
title('Average Fluorescence Intensity', 'FontSize', 28, 'FontWeight', 'bold');
xticklabels(sampleInfo(:, 2));
xtickangle(45); % Rotate x-axis tick labels by 45 degrees
