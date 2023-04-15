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

function ailerons()
    df = read(datadir("exp_raw", "ailerons", "ailerons.data"), DataFrame;
        header=[
            :climbRate, :Sgz, :p, :q, :curPitch, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
            :SeTime1, :SeTime2, :SeTime3, :SeTime4, :SeTime5, :SeTime6, :SeTime7,
            :SeTime8, :SeTime9, :SeTime10, :SeTime11, :SeTime12, :SeTime13, :SeTime14, 
            :diffSeTime1, :diffSeTime2, :diffSeTime3, :diffSeTime4, :diffSeTime5, :diffSeTime6, :diffSeTime7,
            :diffSeTime8, :diffSeTime9, :diffSeTime10, :diffSeTime11, :diffSeTime12, :diffSeTime13, :diffSeTime14, 
            :alpha, :Se, :goal, 
        ]
    )
    return df, :goal
end

function cancer()
    df = read(datadir("exp_raw", "cancer"), DataFrame;
        header=false
    )
    return df, :Column2
end

function compActs()
    df = read(datadir("exp_raw", "compActs", "cpu_act.data"), DataFrame;
        header=[
            :lread, :lwrite, :scall, :sread, :swrite, :fork, :exec, :rchar, :wchar,
            :pgout, :ppgout, :pgfree, :pgscan, :atch, :pgin, :ppgin, :pflt, :vflt,
            :runqsz, :freemem, :freeswap, :usr
        ]
    )
    return df, :usr
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

function elevators()
    df = read(datadir("exp_raw", "elevators", "elevators.data"), DataFrame;
        header=[
            :climbRate, :Sgz, :p, :q, :curRoll, :absRoll, :diffClb, :diffRollRate, :diffDiffClb,
            :SaTime1, :SaTime2, :SaTime3, :SaTime4, :diffSaTime1, :diffSaTime2, :diffSaTime3, :diffSaTime4,
            :Sa, :Goal
        ]
    )
    return df, :Goal
end

function stock()
    df = read(datadir("exp_raw", "stock", "stock.data"), DataFrame;
        delim="\t",
        ignorerepeated=true,
        header=[
            :Company1, :Company2, :Company3, :Company4, :Company5,
            :Company6, :Company7, :Company8, :Company9, :Company10,
        ]
    ) |>
    (X -> coerce(X,
            :Company1 => Continuous,
            :Company2 => Continuous,
            :Company3 => Continuous,
            :Company4 => Continuous,
            :Company5 => Continuous,
            :Company6 => Continuous,
            :Company7 => Continuous,
            :Company8 => Continuous,
            :Company9 => Continuous,
            :Company10 => Continuous
        ))
    return df, :Company10
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
