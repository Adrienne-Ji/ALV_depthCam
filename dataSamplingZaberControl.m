clear all;
clc;
import zaber.motion.ascii.Connection;
import zaber.motion.Units;

% Connection setup
connection = Connection.openSerialPort('COM4');

% Load the waveform data
data = readtable("C:\Users\z5448472\OneDrive - UNSW\Desktop\Recordings6_13\modifiedWaveform_with_home.csv");

% Extract columns
time_points = data.("Time_s_");
pos_U1 = data.("PositionU1_mm_");
pos_U2 = data.("PositionU2_mm_");
pos_U3 = data.("PositionU3_mm_");

% Position limits
absolute_min = 200;
absolute_max = 248;

%%% remember to change log file name!!!!!
% Log file setup with duplicate check
base_name = 'movement_logAWSBatch';
log_file = [base_name '.csv'];

if exist(log_file, 'file')
    serial = 1;
    while exist(sprintf('%s_%d.csv', base_name, serial), 'file')
        serial = serial + 1;
    end
    log_file = sprintf('%s_%d.csv', base_name, serial);
    fprintf('Log file already exists. Saving as: %s\n', log_file);
end

fid = fopen(log_file, 'w');
fprintf(fid, 'Time Stamp,Elapsed Time (s),Position_U1 (mm),Position_U2 (mm),Position_U3 (mm)\n');
fclose(fid);
fid = fopen(log_file, 'a');

% Start the timer
overall_start = tic;

try
    % Enable alerts
    connection.enableAlerts();

    % Detect devices
    deviceList = connection.detectDevices();
    fprintf('Found %d devices.\n', deviceList.length);
    device1 = deviceList(1);
    axis1 = device1.getAxis(1);
    axis2 = device1.getAxis(2);
    axis3 = device1.getAxis(3);

    % Main movement loop
    num_rows = height(data);
    prev_pos1 = pos_U1(1);
    prev_pos2 = pos_U2(1);
    prev_pos3 = pos_U3(1);

    for i = 1:num_rows
        if i == 1 || i == num_rows
            % Set max speed
            axis1.getSettings().set('maxspeed', 15, Units.VELOCITY_MILLIMETRES_PER_SECOND);
            axis2.getSettings().set('maxspeed', 15, Units.VELOCITY_MILLIMETRES_PER_SECOND);
            axis3.getSettings().set('maxspeed', 15, Units.VELOCITY_MILLIMETRES_PER_SECOND);
        else
            axis1.getSettings().set('maxspeed', 15, Units.VELOCITY_MILLIMETRES_PER_SECOND);
            axis2.getSettings().set('maxspeed', 15, Units.VELOCITY_MILLIMETRES_PER_SECOND);
            axis3.getSettings().set('maxspeed', 15, Units.VELOCITY_MILLIMETRES_PER_SECOND);
        end

        % Clamp positions within the valid range
        pos1 = min(max(pos_U1(i), absolute_min), absolute_max);
        pos2 = min(max(pos_U2(i), absolute_min), absolute_max);
        pos3 = min(max(pos_U3(i), absolute_min), absolute_max);

        % Log data before move
        now_time = datestr(datetime, 'yyyy-mm-dd HH:MM:ss.FFF');
        elapsed_time = toc(overall_start);
        log_entry = sprintf('%s,%.6f,%d,%d,%d\n', now_time, elapsed_time, pos1, pos2, pos3);
        fprintf(fid, '%s', log_entry);

        % Move to the target position
        axis1.moveAbsolute(pos1, Units.LENGTH_MILLIMETRES, false);
        axis2.moveAbsolute(pos2, Units.LENGTH_MILLIMETRES, false);
        axis3.moveAbsolute(pos3, Units.LENGTH_MILLIMETRES, false);

        % Initial long pause
        if i == 1
            pause(2);
        else
            pause(0.5);
        end

        % Display progress
        fprintf('Progress: %.2f%%\n', (i / num_rows) * 100);
    end

catch exception
    connection.close();
    fclose(fid);
    rethrow(exception);
end

fclose(fid);

% Close the connection
connection.close();

% Display success message
fprintf('Movement completed. Log saved to: %s\n', log_file);
