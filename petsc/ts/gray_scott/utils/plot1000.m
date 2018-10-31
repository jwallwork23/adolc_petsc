% Plotting script for Gray-Scott problem
%
% Command line options:
% -da_grid_x 1000 -da_grid_y 1000
% -pc_type none
% -ts_max_steps 100 -ts_trajectory_type memory
% -malloc_hbw

fontsize_labels = 14;
fontsize_grid   = 12;
fontname = 'Times';

ncores = [4,16,64];
byhand = [3122.0,814.8,250.0];
%sparse = [,2312.0,654.4];
%matfree = [,,1258.0];

hold off
clf
set(gca,'FontName',fontname)
set(gca,'FontSize',fontsize_grid)
set(gca,'FontSize',fontsize_labels)
semilogx(ncores,byhand,'-+','Markersize',6,'LineWidth',2);
hold on
legend('Hand-coded','Location','NorthWest');
xlabel('Number of cores');
ylabel('Runtime (s)');
title('Gray-Scott problem on KNL with checkpointing')