import os
import sys
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as ani

import utils
import models
import tracks
import trackers

obs_df = None
trk_df = None
tgt_df = None
sen_df = None

def main():    
    # animation()
    # plot2D()
    statistics()


def statistics():
    # obs_list_df = pd.read_csv("obs.csv", index_col=0, parse_dates=True)
    trk_df = pd.read_csv("trk.csv", index_col=0, parse_dates=True)
    tgt_df = pd.read_csv("tgt.csv", index_col=0, parse_dates=True)
    # sen_list_df = pd.read_csv("sen.csv", index_col=0, parse_dates=True)

    scan_id_max = max( [ trk_df.SCAN_ID.max(), tgt_df.SCAN_ID.max() ] ) 
    scan_id_min = min( [ trk_df.SCAN_ID.min(), tgt_df.SCAN_ID.min() ] ) 

    trk_truth_df = pd.DataFrame()

    for i_scan in range(int(scan_id_min), int(scan_id_max)):
        tgt_list = [ models.BaseTarget.from_series(tgt_sr) for _, tgt_sr in tgt_df[tgt_df.SCAN_ID==i_scan].iterrows() ]
        trk_list = [ tracks.BaseTrack.from_series(trk_sr) for _, trk_sr in  trk_df[trk_df.SCAN_ID==i_scan].iterrows() ]
        trk_truth = trackers.TrackerEvaluator.calc_track_truth(tgt_list, trk_list)
        print(trk_truth)


def plot2D():
    obs_df = pd.read_csv("obs.csv", index_col=0, parse_dates=True)
    trk_df = pd.read_csv("trk.csv", index_col=0, parse_dates=True)
    # tgt_df = pd.read_csv("tgt.csv", index_col=0, parse_dates=True)
    # sen_df = pd.read_csv("sen.csv", index_col=0, parse_dates=True)

    mer_df = pd.merge(left=trk_df, right=obs_df, how="left", on="OBS_ID", suffixes=["_TRK", "_OBS"])

    plt.axis("equal")
    plt.grid()

    for sen_id in obs_df["SEN_ID"].unique():

        obs_data = obs_df[obs_df.SEN_ID==sen_id]
        plt.plot(
            obs_data.POSIT_X.values,
            obs_data.POSIT_Y.values,
            marker="D", alpha=.5, linestyle="None", label="obs"
        )

    for trk_id in mer_df["TRK_ID"].unique():

        trk_data = mer_df[mer_df.TRK_ID == trk_id].sort_values("SCAN_ID_TRK")
        trk_data = trk_data[trk_data.OBS_ID!=-1]
        
        plt.plot(
            trk_data.POSIT_X_OBS.values, trk_data.POSIT_Y_OBS.values,
            marker="None", color="r", alpha=1.0, linestyle="solid", label="trk"
        )

        plt.annotate(
            str(int(trk_id)),
            xy=(trk_data.tail(1).POSIT_X_OBS, trk_data.tail(1).POSIT_Y_OBS)
        )

    plt.show()


def animation():
    obs_df = pd.read_csv("obs.csv", index_col=0, parse_dates=True)
    trk_df = pd.read_csv("trk.csv", index_col=0, parse_dates=True)
    tgt_df = pd.read_csv("tgt.csv", index_col=0, parse_dates=True)
    sen_df = pd.read_csv("sen.csv", index_col=0, parse_dates=True)

    scan_id_max = max( [obs_df.SCAN_ID.max(), trk_df.SCAN_ID.max(), tgt_df.SCAN_ID.max() ] ) 
    scan_id_min = min( [obs_df.SCAN_ID.min(), trk_df.SCAN_ID.min(), tgt_df.SCAN_ID.min() ] ) 

    art_list =[]
    fig = plt.figure()
    plt.axis("equal")
    plt.grid()

    obs_art = []
    tgt_art = []
    trk_art = []
    sen_art = []
    pat_art = []

    for i_scan in range(int(scan_id_min), int(scan_id_max)):

        obs_art = plt.plot(
            obs_df[obs_df.SCAN_ID == i_scan].POSIT_X.values,
            obs_df[obs_df.SCAN_ID == i_scan].POSIT_Y.values,
            marker="D", color="g", alpha=.5, linestyle="None", label="obs"
        )

        tgt_art = plt.plot(
            tgt_df[tgt_df.SCAN_ID == i_scan].POSIT_X.values,
            tgt_df[tgt_df.SCAN_ID == i_scan].POSIT_Y.values,
            marker="D", color="b", alpha=.5, linestyle="None", label="tgt"
        )

        trk_art = plt.plot(
            trk_df[trk_df.SCAN_ID == i_scan].POSIT_X.values,
            trk_df[trk_df.SCAN_ID == i_scan].POSIT_Y.values,
            marker="D", color="r", alpha=.5, linestyle="None", label="trk"
        )

        sen_art = plt.plot(
            sen_df[sen_df.SCAN_ID == i_scan].POSIT_X.values,
            sen_df[sen_df.SCAN_ID == i_scan].POSIT_Y.values,
            marker="D", color="g", alpha=.5, linestyle="None", label="sen"
        )
        
        pat_art=[]
        for sen in sen_df[sen_df.SCAN_ID == i_scan].itertuples():
            if sen.SEN_TYPE == "Polar2DSensor":
                pat_art.append(plt.gca().add_patch(pat.Wedge(
                    center=(sen.POSIT_X, sen.POSIT_Y),
                    r=sen.RANGE_MAX,
                    theta1=sen.THETA_MIN*180/cmath.pi,
                    theta2=sen.THETA_MAX*180/cmath.pi,
                    width=sen.RANGE_MAX - sen.RANGE_MIN,
                    color="g",
                    alpha=0.2
                )))
            else:
                raise NotImplementedError

        ax_pos = plt.gca().get_position()
        count = fig.text( ax_pos.x1-0.1, ax_pos.y1-0.05, "count:" + str(i_scan), size = 10 )

        art_list.append( obs_art + trk_art + tgt_art + sen_art + pat_art + [count] )


    _ = ani.ArtistAnimation(fig, art_list, interval=200)
    plt.show()


""" Execute Section """
if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        main()