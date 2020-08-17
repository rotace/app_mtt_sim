import os
import sys
import cmath
import sqlite3
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


class BaseAnalyzer():
    
    @classmethod
    def import_df(cls, tracker, obs_df=None, trk_df=None, sen_df=None, tgt_df=None):
        # add model info
        if tracker:
            obs_df = models.ModelType.add_mdl_info(obs_df, tracker.y_mdl_type())
        return cls(obs_df, trk_df, sen_df, tgt_df)

    @classmethod
    def import_csv(cls, filename="data"):
        obs_df = pd.read_csv(filename+"_obs.csv", index_col=0, parse_dates=True)
        trk_df = pd.read_csv(filename+"_trk.csv", index_col=0, parse_dates=True)
        tgt_df = pd.read_csv(filename+"_tgt.csv", index_col=0, parse_dates=True)
        sen_df = pd.read_csv(filename+"_sen.csv", index_col=0, parse_dates=True)
        return cls(obs_df, trk_df, sen_df, tgt_df)
    
    @classmethod
    def import_db(cls, filename="data"):
        filename+=".db"
        db = sqlite3.connect(filename)
        obs_df = pd.read_sql_query("SELECT * FROM obs", db)
        trk_df = pd.read_sql_query("SELECT * FROM trk", db)
        tgt_df = pd.read_sql_query("SELECT * FROM tgt", db)
        sen_df = pd.read_sql_query("SELECT * FROM sen", db)
        db.close()
        return cls(obs_df, trk_df, sen_df, tgt_df)

    def __init__(self, obs_df, trk_df, sen_df, tgt_df):
        self.obs_df=obs_df
        self.trk_df=trk_df
        self.sen_df=sen_df
        self.tgt_df=tgt_df

    @staticmethod
    def _remove(filename):
        try:
            os.remove(filename)
        except OSError:
            pass

    def _write_df_on_db(self, db):
        self.obs_df.to_sql("obs", db, if_exists="append", index=None)
        self.trk_df.to_sql("trk", db, if_exists="append", index=None)
        self.tgt_df.to_sql("tgt", db, if_exists="append", index=None)
        self.sen_df.to_sql("sen", db, if_exists="append", index=None)

    def export_csv(self, filename="data"):
        self.obs_df.to_csv(filename+"_obs.csv")
        self.trk_df.to_csv(filename+"_trk.csv")
        self.sen_df.to_csv(filename+"_sen.csv")
        self.tgt_df.to_csv(filename+"_tgt.csv")

    def export_db(self, filename="data"):
        filename+=".db"
        self._remove(filename)
        db = sqlite3.connect(filename)
        self._write_df_on_db(db)
        db.close()

    def statistics(self):
        scan_id_max = max( [ self.trk_df.SCAN_ID.max(), self.tgt_df.SCAN_ID.max() ] ) 
        scan_id_min = min( [ self.trk_df.SCAN_ID.min(), self.tgt_df.SCAN_ID.min() ] ) 

        trk_truth_df = pd.DataFrame()

        for i_scan in range(int(scan_id_min), int(scan_id_max)):
            tgt_list = [ models.BaseTarget.from_record(tgt_sr) for _, tgt_sr in self.tgt_df[self.tgt_df.SCAN_ID==i_scan].iterrows() ]
            trk_list = [ tracks.BaseTrack.from_record(trk_sr) for _, trk_sr in  self.trk_df[self.trk_df.SCAN_ID==i_scan].iterrows() ]
            trk_truth = trackers.TrackerEvaluator.calc_track_truth(tgt_list, trk_list)
            print(trk_truth)

    def plot2D(self, filename="data", formats=["plt"]):
        if "db" in formats:
            self._remove(filename)
            db = sqlite3.connect(filename+".db")
        else:
            db = sqlite3.connect(":memory:")

        self._write_df_on_db(db)

        query="""
            CREATE VIEW plot2d_trk_obs AS
            SELECT
                trk.SCAN_ID,
                trk.TRK_ID,
                trk.OBS_ID,
                obs.POSIT_X AS OBS_POSIT_X,
                obs.POSIT_Y AS OBS_POSIT_Y
            FROM trk
            INNER JOIN obs 
            ON
                trk.OBS_ID == obs.OBS_ID AND
                trk.SCAN_ID == obs.SCAN_ID
            ORDER BY
                trk.TRK_ID  ASC,
                trk.SCAN_ID ASC
        """
        db.execute(query)
        trk_obs_df = pd.read_sql_query("SELECT * FROM plot2d_trk_obs", db)

        for trk_id in trk_obs_df.TRK_ID.unique():
            trk_data = trk_obs_df[trk_obs_df.TRK_ID == trk_id]
            plt.plot(
                trk_data.OBS_POSIT_X.values,
                trk_data.OBS_POSIT_Y.values,
                marker="None", color="r", alpha=1.0, linestyle="solid", label="trk"
            )
            plt.annotate(
                str(int(trk_id)),
                xy=(trk_data.tail(1).OBS_POSIT_X, trk_data.tail(1).OBS_POSIT_Y)
            )

        query="""
            CREATE VIEW plot2d_sen_obs AS
            SELECT
                obs.SCAN_ID,
                obs.SEN_ID,
                obs.OBS_ID,
                obs.POSIT_X AS OBS_POSIT_X,
                obs.POSIT_Y AS OBS_POSIT_Y
            FROM obs
            ORDER BY
                obs.SEN_ID  ASC,
                obs.SCAN_ID ASC
        """
        db.execute(query)
        sen_obs_df = pd.read_sql_query("SELECT * FROM plot2d_sen_obs", db)

        for sen_id in sen_obs_df.SEN_ID.unique():
            obs_data = sen_obs_df[sen_obs_df.SEN_ID==sen_id]
            plt.plot(
                obs_data.OBS_POSIT_X.values,
                obs_data.OBS_POSIT_Y.values,
                marker="D", alpha=.5, linestyle="None", label="obs"
            )

        if "csv" in formats:
            trk_obs_df.to_csv(filename+"_plot2d_trk_obs.csv")
            sen_obs_df.to_csv(filename+"_plot2d_sen_obs.csv")
            
        plt.axis("equal")
        plt.grid()
        db.close()

        if "png" in formats:
            plt.savefig(filename+".png")
        if "plt" in formats:
            plt.show()

    def animation(self, interval=200):
        scan_id_max = max( [self.obs_df.SCAN_ID.max(), self.trk_df.SCAN_ID.max(), self.tgt_df.SCAN_ID.max() ] ) 
        scan_id_min = min( [self.obs_df.SCAN_ID.min(), self.trk_df.SCAN_ID.min(), self.tgt_df.SCAN_ID.min() ] ) 

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
                self.obs_df[self.obs_df.SCAN_ID == i_scan].POSIT_X.values,
                self.obs_df[self.obs_df.SCAN_ID == i_scan].POSIT_Y.values,
                marker="D", color="g", alpha=.5, linestyle="None", label="obs"
            )

            tgt_art = plt.plot(
                self.tgt_df[self.tgt_df.SCAN_ID == i_scan].POSIT_X.values,
                self.tgt_df[self.tgt_df.SCAN_ID == i_scan].POSIT_Y.values,
                marker="D", color="b", alpha=.5, linestyle="None", label="tgt"
            )

            trk_art = plt.plot(
                self.trk_df[self.trk_df.SCAN_ID == i_scan].POSIT_X.values,
                self.trk_df[self.trk_df.SCAN_ID == i_scan].POSIT_Y.values,
                marker="D", color="r", alpha=.5, linestyle="None", label="trk"
            )

            sen_art = plt.plot(
                self.sen_df[self.sen_df.SCAN_ID == i_scan].POSIT_X.values,
                self.sen_df[self.sen_df.SCAN_ID == i_scan].POSIT_Y.values,
                marker="D", color="g", alpha=.5, linestyle="None", label="sen"
            )
            
            pat_art=[]
            for sen in self.sen_df[self.sen_df.SCAN_ID == i_scan].itertuples():
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

        _ = ani.ArtistAnimation(fig, art_list, interval=interval)
        plt.show()


def main(): 
    anal = BaseAnalyzer.import_db()
    anal.animation()
    # anal.plot2D(filename="test", formats=["plt", "png", "db", "csv"])
    # anal.statistics()

""" Execute Section """
if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        main()