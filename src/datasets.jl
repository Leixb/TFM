#!/usr/bin/env julia

module DataSets

import DrWatson.datadir
import CSV.read
import DataFrames.DataFrame
using MLJ: coerce, Multiclass, Continuous

function abalone()
    df = read(datadir("exp_raw", "abalone"), DataFrame;
        header=[:Sex, :Length, :Diameter, :Height, :Whole_weight, :Shucked_weight, :Viscera_weight, :Shell_weight, :Rings]
    ) |>
         (X -> coerce(X, :Rings => Continuous))
    return df, :Rings
end

function cpu()
    df = read(datadir("exp_raw", "cpu"), DataFrame;
        header=[:Vendor, :Model, :MYCT, :MMIN, :MMAX, :CACH, :CHMIN, :CHMAX, :PRP, :ERP]
    ) |>
         (X -> coerce(X,
        :Model => Multiclass,
        :Vendor => Multiclass,
        :MYCT => Continuous,
        :MMIN => Continuous,
        :MMAX => Continuous,
        :CACH => Continuous,
        :CHMIN => Continuous,
        :CHMAX => Continuous,
        :PRP => Continuous,
        :ERP => Continuous
    ))
    return df, :ERP
end

function triazines()
    df = read(datadir("exp_raw", "triazines", "triazines.data"), DataFrame;
        header=[
            :p1_polar, :p1_size, :p1_flex, :p1_h_doner, :p1_h_acceptor, :p1_pi_doner, :p1_pi_acceptor, :p1_polarisable, :p1_sigma, :p1_branch,
            :p2_polar, :p2_size, :p2_flex, :p2_h_doner, :p2_h_acceptor, :p2_pi_doner, :p2_pi_acceptor, :p2_polarisable, :p2_sigma, :p2_branch,
            :p3_polar, :p3_size, :p3_flex, :p3_h_doner, :p3_h_acceptor, :p3_pi_doner, :p3_pi_acceptor, :p3_polarisable, :p3_sigma, :p3_branch,
            :p4_polar, :p4_size, :p4_flex, :p4_h_doner, :p4_h_acceptor, :p4_pi_doner, :p4_pi_acceptor, :p4_polarisable, :p4_sigma, :p4_branch,
            :p5_polar, :p5_size, :p5_flex, :p5_h_doner, :p5_h_acceptor, :p5_pi_doner, :p5_pi_acceptor, :p5_polarisable, :p5_sigma, :p5_branch,
            :p6_polar, :p6_size, :p6_flex, :p6_h_doner, :p6_h_acceptor, :p6_pi_doner, :p6_pi_acceptor, :p6_polarisable, :p6_sigma, :p6_branch,
            :activity,
        ]
    ) |>
         (X -> coerce(X, :p2_pi_doner => Continuous))
    return df, :activity
end

end
