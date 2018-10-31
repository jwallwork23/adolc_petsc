% Plotting script for Gray-Scott problem
%
% Command line options:
% -da_grid_x 1000 -da_grid_y 1000
% -pc_type none
% -ts_max_steps 100 -ts_trajectory_type memory
% -malloc_hbw

ncores = [4,16,64];
byhand = [,814.8,250.0];
%sparse = [,,654.4];
%matfree = [,,1258.0];

plot(ncores,byhand),
legend('Hand-coded','Location','NorthWest')
xlabel('Number of cores'),ylabel('Runtime (s)')
title('Gray-Scott problem on KNL with checkpointing')