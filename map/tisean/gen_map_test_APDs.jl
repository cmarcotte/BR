push!(LOAD_PATH,pwd()*"/../")
using map

BCLs = 400.0:-40.0:40.0;
N = 4096;
APDs = zeros(Float64,N,length(BCLs));

for (m,CL) in enumerate(BCLs)
	u = copy(map.u0);
	p = copy(map.p0);
	p[1] = CL;
	APDs[1,m] = f(u,p);
	for n in 2:N
		APDs[n,m] = f(APDs[n-1,m],p);
	end
end

using DelimitedFiles
open("map_APDs.txt", "w") do io
	writedlm(io, APDs)
end
