for Q = [301,401,421,426,441,451];

	close("all");
	fig,axs = plt.subplots(2,1,figsize=(4,4))

	BCL = BCLs[Q];
	APD = APDs[Q];

	axs[1].plot(APD,".k")
	axs[1].set_ylabel("APD\$_n\$")
	axs[1].set_xlabel("\$n\$")

	neigh = -1:8;
	R = reconstruct(APD, 1, 1)
	DLE0 = numericallyapunov(R, 1:5; ntype=FixedMassNeighborhood(2)); 

	for n=1:length(DLE0)
		axs[2].plot([neigh[1],neigh[end]], [DLE0[n],DLE0[n]], "-k"); 
	end
	for n=neigh; 
		try
			DLE = numericallyapunov(R, 1:5; ntype=FixedSizeNeighborhood(2.0^n));  		
			axs[2].plot(n.*ones(length(DLE)), DLE, ".", markersize=8); 
		catch
		end
	end
	axs[2].set_ylabel("\$ \\lambda_1 \$")
	axs[2].set_xlabel("n: ntype=FixedSizeNeighborhood(2.0^n)")

	plt.savefig("$(Q)_lyap_compare.pdf",bbox_inches="tight")

end
