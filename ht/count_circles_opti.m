function [c1,rad1,c2,rad2] = test(path_in, sb, tb, sd, td, type)
im = imread(path_in);
c1=0;
c2=0;
rad1=0;
rad2=0;
if type == -1 | type == 1
    [c1,rad1]=imfindcircles(im,[16 17],'EdgeThreshold', tb, 'Sensitivity', sb,'ObjectPolarity','bright', 'Method', 'twoStage');
end
if type == -1 | type == 0
    [c2,rad2]=imfindcircles(im,[16 17],'EdgeThreshold', td, 'Sensitivity', sd,'ObjectPolarity','dark', 'Method', 'twoStage');
end