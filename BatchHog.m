function BatchHog
    d = str2num(input('please input id range :', 's'));
    s = d(1);
    e = d(2);
    mkdir('hog_feature');
    variance = zeros(e-s, 2);
    num = 0;
    for i= s:e
        dir = ['exemplars' filesep num2str(i) '.jpg'];
        if ~exist(dir, 'file')
            continue;
        end
        img = imread(dir);
        [window_var, cell_var, grad] = calVar(img, 16, 4);
        num = num+1;
        variance(num, :) = [window_var cell_var];
        disp([num2str(i) '- window_var = ' num2str(window_var,'%.2f') ', cell_var = ' num2str(cell_var,'%.2f')]);
        dlmwrite(['hog_feature' filesep num2str(i) '_src_u.txt'], grad(:,:,1));
        dlmwrite(['hog_feature' filesep num2str(i) '_src_v.txt'], grad(:,:,2));
        print(gcf,'-dpng',['hog_feature', filesep num2str(i), '_hog.png']);
    end
    dlmwrite(['hog_feature' filesep 'variance.txt'], variance);
end

function [sum_var, cell_var, grad] = calVar(img, cellSize, windowSize)
    figure;
    imshow(img);
    hold on;
    [~, visualization] = extractHOGFeatures(img, 'CellSize', [cellSize cellSize], 'BlockSize', [2 2], 'NumBins', 9);
    plot(visualization);
    
    img = rgb2gray(img);
    [rows cols channels] = size(img);
    loc = [];
    nums = 0;
    for j = 1:cellSize:cols
        for i = 1:cellSize:rows
            nums = nums+1;
            loc(nums, :) = [j, i];
        end
    end
    [hog, validPoints, visualization] = extractHOGFeatures(img, loc, 'CellSize', [cellSize cellSize], 'BlockSize', [2 2], 'NumBins', 9);
    nums = size(validPoints,1);
    rownum = floor(rows/cellSize)-2+1;
    colnum = floor(cols/cellSize)-2+1;
    domainOri = zeros(rownum,colnum, 2);
    perpOri = zeros(rownum,colnum, 2);
    cell_var = 0;
    nCellNull = 0;
    for i = 1:nums
        x = validPoints(i,1)/10;
        y = validPoints(i,2)/10;
        h = (hog(i, 1:9));
        initData = zeros(36, 2);
        for ang = 0:17
            dx = h(mod(ang, 9)+1)*cos((ang-4)*pi/9);
            dy = h(mod(ang, 9)+1)*sin((ang-4)*pi/9);
            initData(ang+1, :) = [dx dy];
        end
        initData(19:end, :) = -initData(1:18, :);
        [u s v] = svd(initData'*initData);
        domainOri( mod((i-1),rownum)+1, floor((i-1)/rownum)+1, :) = s(1,1)*[v(1,1) -v(2,1)];
        perpOri(mod((i-1),rownum)+1, floor((i-1)/rownum)+1, :) = s(2,2)*[v(1,2) -v(2,2)];
        if s(1,1) < 1e-5
            nCellNull = nCellNull+1;
        else
            cell_var = cell_var+s(2,2)/s(1,1);
        end
    end

    u = imresize(domainOri(:,:,1), [rows cols], 'lanczos3');
    v = imresize(domainOri(:,:,2), [rows cols], 'lanczos3');
    grad = zeros(rows, cols, 2);
    grad(:,:,1) = u;
    grad(:,:,2) = v;

    cell_var = cell_var/(nums-nCellNull);
    sum_var = 0;
    num = 0;
    data = zeros(4*windowSize*windowSize, 2);
    for i = 1:rownum-windowSize+1
        for j = 1:colnum-windowSize+1
            data(1:windowSize*windowSize, :) = reshape(domainOri(i:i+windowSize-1, j:j+windowSize-1, :), windowSize*windowSize, 2);
            data(windowSize*windowSize+1:2*windowSize*windowSize, :) = reshape(perpOri(i:i+windowSize-1, j:j+windowSize-1, :), windowSize*windowSize, 2);
            data(2*windowSize*windowSize+1:4*windowSize*windowSize, :) = -data(1:2*windowSize*windowSize, :);
            [u s v] = svd(data'*data);
            if s(1,1) < 1e-5
                variance = 0;
                num = num+1;
            else
                variance = s(2,2)/s(1,1);
            end
            sum_var = sum_var + variance;
        end
    end
    sum_var = sum_var/((rownum-windowSize+1)*(colnum-windowSize+1)-num);
end