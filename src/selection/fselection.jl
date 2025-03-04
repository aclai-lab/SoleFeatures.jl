for i in 1:size(a[1],2)
    if a[1][:,i] != b[1][:,i]
        println(i)
    end
end

# valid_X[:, 1] = [-0.531415, -0.493256, -0.536751, -0.57022, -0.663721, 