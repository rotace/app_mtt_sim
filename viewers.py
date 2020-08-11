import os
import sys
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as ani


def main():
    obs_df = pd.read_csv("obs.csv")
    trk_df = pd.read_csv("trk.csv")
    tgt_df = pd.read_csv("tgt.csv")
    sen_df = pd.read_csv("sen.csv")

    scan_id_max = max( [obs_df["SCAN_ID"].max(), trk_df["SCAN_ID"].max(), tgt_df["SCAN_ID"].max() ] ) 
    scan_id_min = min( [obs_df["SCAN_ID"].min(), trk_df["SCAN_ID"].min(), tgt_df["SCAN_ID"].min() ] ) 

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