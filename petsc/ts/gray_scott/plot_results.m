function []=plot_results(n)
% Plotting script for Gray-Scott problem
%
% To run, activate Octave and call plot_results(N), where N is the number of grid points used in
% each direction.
%
% PETSc command line options used:
% -da_grid_x N -da_grid_y N
% -pc_type none
% -ts_max_steps 100 -ts_trajectory_type memory
% -malloc_hbw

clf
set(gca,'FontName','Times')
set(gca,'FontSize',12)
set(gca,'FontSize',14)
N = int2str(n);

byhand = load(strcat('data/byhand',N,'.txt'));
ncores = byhand(:,1);
idx = ~isnan(byhand(:,2));
loglog(ncores(idx,1),byhand(idx,2),'-+');
hold on

if n == 65
    full = load(strcat('data/full',N,'.txt'));
    idx = ~isnan(full(:,2));
    loglog(ncores(idx,1),full(idx,2),'-^');
end

sparse = load(strcat('data/sparse',N,'.txt'));
idx = ~isnan(sparse(:,2));
loglog(ncores(idx,1),sparse(idx,2),'-o');

matfree = load(strcat('data/matfree',N,'.txt'));
idx = ~isnan(matfree(:,2));
loglog(ncores(idx,1),matfree(idx,2),'-*');

if n == 65
    legend('Hand-coded','ADOL-C','Sparse ADOL-C','Matrix-free ADOL-C','Location','NorthEast');
else
    %loglog(ncores_byhand,byhand,'-+',ncores_sparse,sparse,'-o',ncores_matfree,matfree,'-*','Markersize',6,'LineWidth',2);
    legend('Hand-coded','Sparse ADOL-C','Matrix-free ADOL-C','Location','NorthEast');
end
xlabel('Number of cores');
ylabel('Runtime (s)');
title(strcat('Gray-Scott problem on ',{' '},N,'x',N,' grid with checkpointing'))
saveas(gcf,outfile = strcat('plots/res',N,'.png'));

end
