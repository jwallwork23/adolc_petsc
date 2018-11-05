function []=plot_results(N)
% Plotting script for Gray-Scott problem
%
% Command line options:
% -da_grid_x N -da_grid_y N
% -pc_type none
% -ts_max_steps 100 -ts_trajectory_type memory
% -malloc_hbw

fontsize_labels = 14;
fontsize_grid   = 12;
fontname = 'Times';

ncores = [4,16,64];
N = int2str(N);

fbyhand = fopen(strcat('data/byhand',N,'.txt'), 'r');
byhand = fscanf(fbyhand,'%f');
fclose(fbyhand);

fsparse = fopen(strcat('data/sparse',N,'.txt'), 'r');
sparse = fscanf(fsparse,'%f');
fclose(fsparse);

fmatfree = fopen(strcat('data/matfree',N,'.txt'), 'r');
matfree = fscanf(fmatfree,'%f');
fclose(fmatfree);

hold off
clf
set(gca,'FontName',fontname)
set(gca,'FontSize',fontsize_grid)
set(gca,'FontSize',fontsize_labels)
semilogx(ncores,byhand,'-+',ncores,sparse,'-o',ncores,matfree,'-*','Markersize',6,'LineWidth',2);
hold on
legend('Hand-coded','Sparse ADOL-C','Matrix-free ADOL-C','Location','NorthEast');
xlabel('Number of cores');
ylabel('Runtime (s)');
title('Gray-Scott problem on KNL with checkpointing')
saveas(gcf,outfile = strcat('plots/res',N,'.png'));

end