using DataFrames
using CSV
using Plots
using Distributions
using StatsPlots
using HypothesisTests
using PyCall
using IterTools

const shapiro = pyimport("scipy.stats").shapiro
const dagostino = pyimport("scipy.stats").normaltest
const kstest = pyimport("scipy.stats").kstest
const lilliefors = pyimport("statsmodels.stats.diagnostic").lilliefors
# https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
const pymultipletests = pyimport("statsmodels.stats.multitest").multipletests

function extract_descriptor(str::AbstractString)
	m = match(r".*-(.*)$", str)
	return isnothing(m) ? nothing : m[1]
end

function extract_match(str::AbstractString, expr::AbstractString)
	m = match(Regex(".*($expr).*\$"), str)
	return isnothing(m) ? nothing : m[1]
end
function extract_sensor(str::AbstractString)
	m = match(r".*(S[0-9]+).*$", str)
	return isnothing(m) ? nothing : m[1]
end
function extract_band(str::AbstractString)
	m = match(r".*(B[0-9]+).*$", str)
	return isnothing(m) ? nothing : m[1]
end
function extract_sensor_band(str::AbstractString)
	m = match(r".*([BS][0-9]+).*([BS][0-9]+).*$", str)
	return isnothing(m) ? nothing : (m[1], m[2])
end

function group_pvals_by_descriptor(df::AbstractDataFrame)
	descriptors = unique(extract_descriptor.(names(df)))

	pvals = Vector{Vector{Float64}}(undef, length(descriptors))
	Threads.@threads for (i, d) in collect(enumerate(descriptors))
		pvals[i] = Matrix(df[:,filter(endswith(d), names(df))])[:]
	end

	return pvals
end
function group_pvals_by_descriptor(csv::AbstractString)
	return group_pvals_by_descriptor(CSV.read(csv, DataFrame)[:,2:end])
end

"""

## METHODS

* `:nocorrection`: no correction
* `:bonferroni`: one-step correction
* `Sumbol("holm-sidak")`: one-step correction
* `:hs`: step down method using Sidak adjustments
* `:holm`: step-down method using Bonferroni adjustments
* `Symbol("simes-hochberg")`: step-up method (independent)
* `:hommel`: closed method based on Simes tests (non-negative)
* `:fdr_bh`: Benjamini/Hochberg (non-negative)
* `:fdr_by`: Benjamini/Yekutieli (negative)
* `:fdr_tsbh`: two stage fdr correction (non-negative)
* `:fdr_tsbky`: two stage fdr correction (non-negative)

See [`Controlling_procedures`](https://en.wikipedia.org/wiki/Family-wise_error_rate#Controlling_procedures)
for more info and the [python package documentation](https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html).
"""
function multipletests(
	pvals::AbstractVector{<:Real};
	alpha::AbstractFloat = 0.05,
	method::Symbol = Symbol("holm-sidak"),
	issorted::Bool = false,
	returnsorted::Bool = false
)
	@assert 0 ≤ alpha ≤ 1 "`alpha` has to be in range [0, 1]"
	@assert count(x -> 0 ≤ x ≤ 1, pvals) == length(pvals) "All `pvals` has to be in range [0, 1]"
	possible_methods = Symbol.(["nocorrection", "bonferroni", "sidak", "holm-sidak", "holm",
		"simes-hochberg", "hommel", "fdr_bh", "fdr_by", "fdr_tsbh", "fdr_tsbky"])
	if method == :nocorrection
		# no correction
		return (pvals .≤ alpha, pvals, alpha, alpha)
	else
		return pymultipletests(pvals, alpha, string(method), issorted, returnsorted)
	end
end
function multipletests(
	pvals::AbstractDataFrame,
	separate::Union{Symbol,Function} = :all;
	kwargs...
)
	allowed_separates = [:all, :bycol, :byrow]
	@assert isa(separate, Function) || separate in allowed_separates "`separate` can be $(join(string.(":", allowed_separates), ", ")) or a Function: passed $separate"

	if separate == :all
		return multipletests(Matrix(pvals)[:]; kwargs...)
	elseif separate == :bycol
		return [multipletests(pvals[:,i]; kwargs...) for i in 1:ncol(pvals)]
	elseif separate == :byrow
		return [multipletests(collect(pvals[i,:]); kwargs...) for i in 1:nrow(pvals)]
	elseif isa(separate, Function)
		return [multipletests(pv; kwargs...) for pv in separate(pvals)]
	end
end
function multipletests(
	pvals::AbstractString,
	separate::Union{Symbol,Function} = :all;
	kwargs...
)
	return multipletests(CSV.read(pvals, DataFrame)[:,2:end], separate; kwargs...)
end

norm_pop(population) = fit(Normal, population)
# norm_pop(population) = Normal(mean(population), std(population))

function normality_shapiro(population::AbstractVector{<:Real})::Bool
	length(population) == 0 && return false;

	stat, p =
		try
			shapiro(population)
		catch ex
			return false
		end

	return p > 0.05
end
function normality_dagostino(population::AbstractVector{<:Real})::Bool
	length(population) == 0 && return false;

	stat, p =
		try
			dagostino(population)
		catch ex
			return false
		end

	return p > 0.05
end
function normality_lilliefors(population::AbstractVector{<:Real})::Bool
	length(population) == 0 && return false;

	stat, p =
		try
			lilliefors(population)
		catch ex
			return false
		end

	return p > 0.05
end
function normality_anderson_darling(population::AbstractVector{<:Real})::Bool
	length(population) == 0 && return false;

	res = HypothesisTests.OneSampleADTest(population, norm_pop(population))

	return pvalue(res) > 0.05
end
function normality_exact_ks(population::AbstractVector{<:Real})::Bool
	length(population) == 0 && return false;

	res = ExactOneSampleKSTest(population, norm_pop(population))

	return pvalue(res) > 0.05
end
function normality_approx_ks(population::AbstractVector{<:Real})::Bool
	length(population) == 0 && return false;

	res = ApproximateOneSampleKSTest(population, norm_pop(population))

	return pvalue(res) > 0.05
end

function statistical_analysis(
	input_file::AbstractString;
	use_only_classes::Union{<:AbstractVector{<:AbstractString},Nothing} = nothing,
	savefigs::Bool = true,
	saveplots::Bool = false,
	file_prefix::AbstractString = "./",
	normality_tests::AbstractVector = [
		# normality_shapiro,
		# normality_dagostino,
		normality_lilliefors,
		# normality_anderson_darling,
		# normality_exact_ks,
		normality_approx_ks
	],
	param_tests::AbstractVector = [
		HypothesisTests.EqualVarianceZTest,
		HypothesisTests.UnequalVarianceZTest,
		HypothesisTests.EqualVarianceTTest,
		HypothesisTests.UnequalVarianceTTest
	],
	non_param_tests::AbstractVector = [
		HypothesisTests.MannWhitneyUTest,
		HypothesisTests.ApproximateTwoSampleKSTest,
		# HypothesisTests.SignedRankTest
	],
	export_csv::Bool = false,
	silent::Bool = false
)
	function parse_vector_of_number(vector::AbstractString)::Vector{Float64}
		return parse.(Float64, strip.(split(lstrip(rstrip(strip(vector), ']'), '['), ',')))
	end

	_println = silent ? (s...) -> nothing : (s...) -> println(s...)

	function _savefig(plt, plot_name)
		tot_name_no_ext = "$(joinpath(file_prefix, "statistical_analysis"))/$((plot_name == "" ? "" : "-$(plot_name)"))"
		if savefigs || saveplots
			mkpath(dirname(tot_name_no_ext))
		end
		if savefigs
			savefig(plt, "$(tot_name_no_ext).png")
		end
		if saveplots
			serialize("$(tot_name_no_ext).plot.jld", plt)
		end
	end

	if savefigs || saveplots
		mkpath(joinpath(file_prefix, "statistical_analysis"))
	end

	df = CSV.read(input_file, DataFrame)

	class_names =
		if isnothing(use_only_classes)
			df[:,:class_names]
		else
			filter(x -> x in use_only_classes, df[:,:class_names])
		end

	normality_csv =
		if export_csv
			DataFrame(:classnames => class_names)
		end
	pvalue_csv =
		if export_csv
			DataFrame(:classnames => join.(IterTools.subsets(class_names, 2), "-vs-"))
		end

	for feature in filter(x -> x != "class_names", names(df))
		_println("● FEATURE: ", feature)
		class_serie_is_normal = []
		series = []

		# perform normality test for each class for the current feature
		for i_c in 1:nrow(df)
			serie = parse_vector_of_number(df[i_c,feature])
			push!(series, serie)

			# TODO: maybe weighted???
			push!(class_serie_is_normal, length(findall([nt(filter(x -> !isnan(x), serie)) for nt in normality_tests])) >= (length(normality_tests) * 0.5))
		end

		if export_csv
			insertcols!(normality_csv, feature => class_serie_is_normal)
		end

		class_couple_iter = 1
		curr_pvals = []
		for (c1, c2) in IterTools.subsets(class_names, 2)
			_println("   └─ ", c1, " vs ", c2)
			c1_index = findfirst(isequal(string(c1)), df[:, :class_names])
			c2_index = findfirst(isequal(string(c2)), df[:, :class_names])

			s1 = filter(x -> !isnan(x), series[c1_index])
			s2 = filter(x -> !isnan(x), series[c2_index])

			if length(s1) == 0 || length(s2) == 0
				println("Skipping $(feature) with classes $(c1) and $(c2) because at least one of the series was empty")
				continue
			end

			density(s1, title = "$(feature) distributions", label = string(c1, " (norm: $((class_serie_is_normal[c1_index] ? "y" : "n")))"))
			density!(s2, title = "$(feature) distributions", label = string(c2, " (norm: $((class_serie_is_normal[c2_index] ? "y" : "n")))"))

			txt(s::AbstractString) = Plots.text(s, 10, :dark; halign = :left, valign = :top)
			left = Plots.xlims()[1] + ((Plots.xlims()[2] - Plots.xlims()[1]) * 0.05)
			top = Plots.ylims()[2] - ((Plots.ylims()[2] - Plots.ylims()[1]) * 0.05)

			tests_to_perform =
				if false in class_serie_is_normal[[c1_index, c2_index]]
					non_param_tests
				else
					param_tests
				end

			annotation = ""
			for t in tests_to_perform
				stats = t(s1, s2)
				pv = pvalue(stats)
				short_pv = round(pv; digits = 3)
				_println("       └─ ", nameof(t), " p-value: ", short_pv)
				annotation *= string(nameof(t), " p-value: ", short_pv, "\n")

				# NOTE: will be saved always first performed test
				if export_csv && length(curr_pvals) < class_couple_iter
					push!(curr_pvals, pv)
				end
			end

			_savefig(annotate!(left, top, txt(annotation)), "$(c1)_vs_$(c2)/$(feature)_distributions")

			class_couple_iter += 1
		end

		if export_csv
			insertcols!(pvalue_csv, feature => curr_pvals)
		end
	end

	if export_csv
		mkpath(joinpath(file_prefix, "statistical_analysis"))
		CSV.write("$(joinpath(file_prefix, "statistical_analysis", "normality.csv"))", normality_csv)
		CSV.write("$(joinpath(file_prefix, "statistical_analysis", "pvalue.csv"))", pvalue_csv)
	end

	return nothing
end
