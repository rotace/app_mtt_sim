import os
from sensors import BaseSensor
import sys
import fire
import cmath
import pathlib
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
    def import_df(cls, obs_df=None, trk_df=None, sen_df=None, tgt_df=None):
        return cls(obs_df, trk_df, sen_df, tgt_df)

    @classmethod
    def import_csv(cls, fpath="data.xxx"):
        p=pathlib.Path(fpath)
        obs_df = pd.read_csv(str(p.with_name(p.stem+"_obs.csv")), index_col=0, parse_dates=True)
        trk_df = pd.read_csv(str(p.with_name(p.stem+"_trk.csv")), index_col=0, parse_dates=True)
        tgt_df = pd.read_csv(str(p.with_name(p.stem+"_tgt.csv")), index_col=0, parse_dates=True)
        sen_df = pd.read_csv(str(p.with_name(p.stem+"_sen.csv")), index_col=0, parse_dates=True)
        return cls(obs_df, trk_df, sen_df, tgt_df)
    
    @classmethod
    def import_db(cls, fpath="data.xxx"):
        p=pathlib.Path(fpath)
        db = sqlite3.connect(str(p.with_suffix(".db")))
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
    def _remove(fpath):
        try:
            os.remove(fpath)
        except OSError:
            pass

    def _write_df_on_db(self, db):
        self.obs_df.to_sql("obs", db, if_exists="append", index=None)
        self.trk_df.to_sql("trk", db, if_exists="append", index=None)
        self.tgt_df.to_sql("tgt", db, if_exists="append", index=None)
        self.sen_df.to_sql("sen", db, if_exists="append", index=None)

    def export_csv(self, fpath="data.xxx"):
        p=pathlib.Path(fpath)
        self.obs_df.to_csv(str(p.with_name(p.stem+"_obs.csv")))
        self.trk_df.to_csv(str(p.with_name(p.stem+"_trk.csv")))
        self.sen_df.to_csv(str(p.with_name(p.stem+"_sen.csv")))
        self.tgt_df.to_csv(str(p.with_name(p.stem+"_tgt.csv")))

    def export_db(self, fpath="data.xxx"):
        p=pathlib.Path(fpath)
        self._remove(str(p.with_suffix(".db")))
        db = sqlite3.connect(str(p.with_suffix(".db")))
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

    def _pre_plot(self, fpath, formats):
        p=pathlib.Path(fpath)
        if "db" in formats:
            self._remove(str(p.with_suffix(".db")))
            db = sqlite3.connect(str(p.with_suffix(".db")))
        else:
            db = sqlite3.connect(":memory:")

        self._write_df_on_db(db)
        return db

    def _post_plot(self, fpath, formats):
        p=pathlib.Path(fpath)
        plt.grid()
        if "png" in formats:
            plt.savefig(str(p.with_name(p.stem+".png")))
        if "plt" in formats:
            plt.show()

    def plot_score(self, fpath="data.xxx", formats=["plt"]):
        db = self._pre_plot(fpath, formats)

        query="""
            CREATE VIEW plot2d_trk_scr AS
            SELECT
                trk.SCAN_ID,
                trk.TRK_ID,
                trk.SCORE
            FROM trk
            ORDER BY
                trk.TRK_ID  ASC,
                trk.SCAN_ID ASC
        """
        db.execute(query)
        trk_scr_df = pd.read_sql_query("SELECT * FROM plot2d_trk_scr", db)

        for trk_id in trk_scr_df.TRK_ID.unique():
            trk_data = trk_scr_df[trk_scr_df.TRK_ID == trk_id]
            plt.plot(
                trk_data.SCAN_ID.values,
                trk_data.SCORE.values,
                marker="None", alpha=1.0, linestyle="solid", label="trk"
            )
            plt.annotate(
                str(int(trk_id)),
                xy=(trk_data.tail(1).SCAN_ID, trk_data.tail(1).SCORE)
            )

        db.close()
        if "csv" in formats:
            p=pathlib.Path(fpath)
            trk_scr_df.to_csv(str(p.with_name(p.stem+"_plot2d_trk_scr.csv")))
        self._post_plot(fpath, formats)

    def plot2D(self, fpath="data.xxx", formats=["plt"], is_ellipse_enabled=False):
        db = self._pre_plot(fpath, formats)

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
                obs.POSIT_Y AS OBS_POSIT_Y,
                obs.R00,
                obs.R01,
                obs.R11
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
            if is_ellipse_enabled:
                for obs in obs_data.itertuples():
                    cov = np.array([obs.R00, obs.R01, obs.R01, obs.R11]).reshape((2,2))
                    width, height, theta = utils.calc_confidence_ellipse(cov)
                    plt.gca().add_patch(pat.Ellipse(
                        xy=(obs.OBS_POSIT_X, obs.OBS_POSIT_Y),
                        width=width, height=height, angle=np.degrees(theta), color="c", alpha=.2
                    ))

        db.close()
        plt.axis("equal")
        if "csv" in formats:
            p=pathlib.Path(fpath)
            trk_obs_df.to_csv(str(p.with_name(p.stem+"_plot2d_trk_obs.csv")))
            sen_obs_df.to_csv(str(p.with_name(p.stem+"_plot2d_sen_obs.csv")))
        self._post_plot(fpath, formats)

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
            count = fig.text( ax_pos.x0+0.05, ax_pos.y1-0.05, "count:" + str(i_scan), size = 10 )

            art_list.append( obs_art + trk_art + tgt_art + sen_art + pat_art + [count] )

        _ = ani.ArtistAnimation(fig, art_list, interval=interval)
        plt.show()


def create_format_args(plt=False, png=False, db=False, csv=False):
    formats = []
    if plt:
        formats.append("plt")
    if png:
        formats.append("png")
    if db:
        formats.append("db")
    if csv:
        formats.append("csv")
    return formats

class Worker:
    """
    HOW TO USE
    ex1)
    $ python analyzers.py --help
    ex2)
    $ python analyzers.py plot --help
    ex3)
    $ python analyzers.py plot data
    ex4)
    $ python analyzers.py plot -fpath=data
    ex5)
    $ python analyzers.py plot -fpath=data -png -csv
    """
    def anime(self, fpath):
        anal = BaseAnalyzer.import_db(fpath)
        anal.animation()

    def plot(self, fpath, plt=True, png=False, db=False, csv=False):
        formats=create_format_args(plt, png, db, csv)
        anal = BaseAnalyzer.import_db(fpath)
        anal.plot2D(fpath=fpath, formats=formats)

    def score(self, fpath, plt=True, png=False, db=False, csv=False):
        formats=create_format_args(plt, png, db, csv)
        anal = BaseAnalyzer.import_db(fpath)
        anal.plot_score(fpath=fpath, formats=formats)

    def stat(self, fpath):
        anal = BaseAnalyzer.import_db(fpath)
        anal.statistics()


""" Execute Section """
if __name__ == '__main__':
    if (sys.flags.interactive != 1):
        worker = Worker()
        fire.Fire(worker)