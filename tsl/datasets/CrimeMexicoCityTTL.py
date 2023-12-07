
import os

import numpy as np
import pandas as pd
import kglab
import json
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely import wkt

from tsl import logger

from ..ops.similarities import gaussian_kernel
from ..utils import download_url, extract_zip
from .prototypes import DatetimeDataset


class CrimeMexicoCityTTL(DatetimeDataset):
    r"""Traffic readings collected from 207 loop detectors on
    highways in Los Angeles County, aggregated in 5 minutes intervals over four
    months between March 2012 and June 2012.
    
    Registro de diferentes denuncias hechas por ciudadanos en la ciudad de Mexico
    del anio [] al anio []. Los datos provienen de [] pero estan convertidos en TTL
    para crear un grafo de conocimiento. En esta clase se hace uso de SPARQL para
    construir el dataset a partir del grafo de conocimiento guardado en el archivo TTL.

    Dataset information:
        + Time steps: ??
        + Nodes: ??
        + Channels: ??
        + Sampling rate: ?? minutes
        + Missing values: ??%

    Static attributes:
        + :obj:`dist`: :math:`N \times N` matrix of node pairwise distances.

    Los atributos de arriba se van a tener que platicar muy bien entre nosotros :)

    """
    url = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"
    
    print(f">>> CURRENT WORK DIRECTORY: [{os.getcwd()}]")

    similarity_options = {'distance'} # O que vamos a usar como medida de similitud para enlazar los nodos?

    def __init__(self, root=None, impute_zeros=True, freq=None, geo_detail="alcaldia"):
        """
            Function that loads preprocesed TTL files into TSL
            geo_detail: ["alcaldia", "ageb", "alcaldia_femenina", "alcaldia_masculina", "ageb_femenina", "ageb_masculina", "alcaldia_nonzeroentries", "alcaldia_femenina_nonzeroentries", "alcaldia_masculina_nonzeroentries"]
        """
        # set root path
        self.root = root
        
        list_detail = ["alcaldia", "ageb", "alcaldia_femenina", "alcaldia_masculina", "ageb_femenina", "ageb_masculina", "alcaldia_nonzeroentries", "alcaldia_femenina_nonzeroentries", "alcaldia_masculina_nonzeroentries"]
        if geo_detail.lower() not in list_detail:
            raise ValueError('Desired crime not in list: "%s"' % (geo_detail))
            
        self.geo_detail = geo_detail.lower()
        
        df, dist, mask = self.load(impute_zeros=impute_zeros)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="CrimeMexicoCityTTL")

        print("dist shape and dist: ")
        print(dist.shape)
        print(dist)
        self.add_covariate('dist', dist, pattern='n n')

    @property
    def raw_file_names(self):
        return [
#             'metr_la.h5', 'distances_la.csv', 'sensor_locations_la.csv',
#             'sensor_ids_la.txt'
            '/content/tsl/tsl/datasets/raw_files_to_remove/testind3.ttl'
        ]

    # Decoradores para indicar que la clase debe de contar con tales archivos
    @property
    def required_file_names(self):
        return ['metr_la.h5', 'metr_la_dist.npy']

    # Este metodo lo podemos dejar asi con el fin de mantener la consistencia con la clase heredada
    # Segun la doc, esto es lo que debe de hacer:
    # Downloads dataset’s files to the self.root_dir folder.
    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)

    # Este metodo lo podemos dejar asi con el fin de mantener la consistencia con la clase heredada
    # Segun la doc, esto es lo que debe de hacer:
    # Eventually build the dataset from raw data to self.root_dir folder.
    def build(self, aggregation_level="alcaldia") -> None:
        self.maybe_download()
        # Build distance matrix
        logger.info('Building distance matrix...')
        raw_dist_path = os.path.join(self.root_dir, 'distances_la.csv')
        distances = pd.read_csv(raw_dist_path)
        ids_path = os.path.join(self.root_dir, 'sensor_ids_la.txt')
        with open(ids_path) as f:
            ids = f.read().strip().split(',')
        num_sensors = len(ids)
        dist = np.ones((num_sensors, num_sensors), dtype=np.float32) * np.inf
        # Builds sensor id to index map.
        sensor_to_ind = {int(sensor_id): i for i, sensor_id in enumerate(ids)}
        # Fills cells in the matrix with distances.
        for row in distances.values:
            if row[0] not in sensor_to_ind or row[1] not in sensor_to_ind:
                continue
            dist[sensor_to_ind[row[0]], sensor_to_ind[row[1]]] = row[2]
        # Save to built directory
        path = os.path.join(self.root_dir, 'metr_la_dist.npy')
        np.save(path, dist)
        # Remove raw data
        self.clean_downloads()

    def load(self, impute_zeros=True):
        
        if self.geo_detail == 'alcaldia':
            df = pd.read_csv('https://drive.google.com/uc?id=1-870PZIVuo-sisabqJiXB3P5yGAQNdWJ')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1--UgH91lTa01GDNyu_nJonznaDbYSI-K')
            dist = np.load(file.name)
        elif self.geo_detail == 'ageb':
            df = pd.read_csv('https://drive.google.com/uc?id=1-CnS8-dxlK2LlSdHc6pFtyV6if9ANcP9')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-4ZHYv49gj0DCck7o207kmI6_7jtC4Eb')
            dist = np.load(file.name)
        elif self.geo_detail == 'alcaldia_femenina':
            df = pd.read_csv('https://drive.google.com/uc?id=1buvfaT1_51Eor6V_xACDkIXMaQWNe7xd')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-BBnrv1dfV1vobfToqaMYtAtgYIGdiUX')
            dist = np.load(file.name)
        elif self.geo_detail == 'alcaldia_masculina':
            df = pd.read_csv('https://drive.google.com/uc?id=1-5qKj5igmJmH2ubNN7IGTevS3wUx_jbw')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-1xwmPnl7x43IjRdjKEr5inpVYAU4gKT')
            dist = np.load(file.name)
        elif self.geo_detail == 'ageb_femenina':
            df = pd.read_csv('https://drive.google.com/uc?id=1-IKeHkP1id317Tvv_pVBZNSt9kmGeNjF')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-HzeNPohSTkOtAVVGUy-hyoeDKd2o2U1')
            dist = np.load(file.name)
        elif self.geo_detail == 'ageb_masculina':
            df = pd.read_csv('https://drive.google.com/uc?id=1-G8-nl7SXM6OwGNdI1sD3fHjvyL6Yqpa')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-FXQBe9eZHxcGyUBGRXdUYSxGL6PM2yH')
            dist = np.load(file.name)

        elif self.geo_detail == 'alcaldia_nonzeroentries':
            df = pd.read_csv('https://drive.google.com/uc?id=1-BMLbAFgxW1hVvgQ57FemevfgrcqnFKx')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-7T8HJ062JdnD7yfl3OUY0us2hp3-0ua')
            dist = np.load(file.name)
        elif self.geo_detail == 'alcaldia_femenina_nonzeroentries':
            df = pd.read_csv('https://drive.google.com/uc?id=1-Coc57nfnihAV61S48Q-wAfegh1_L6Wd')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-T7QTiRV80eS71wIc6NzFOhDGZ3pd5eP')
            dist = np.load(file.name)
        elif self.geo_detail == 'alcaldia_masculina_nonzeroentries':
            df = pd.read_csv('https://drive.google.com/uc?id=1-SjwBsAIqVTRi-Dt4GqIlHY7EhkmGObn')
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            file = np.DataSource().open('https://drive.google.com/uc?id=1-MCOIb4W92EabmF5jxrS76wPruwOf7dd')
            dist = np.load(file.name)
            
        mask = (df.values != 0.).astype('uint8')
        if impute_zeros:
            df = df.replace(to_replace=0., method='ffill')
            
        return df, dist, mask
    
    def compute_similarity(self, method: str, **kwargs):
        # De acuerdo con el atributo similarity_options declarado al inicio de la clase,
        # se debera implementar el tantos metodos como se tengan definidos en tal atributo
        # con el fin de objetener una matriz de  similitud para luego poder crear la matriz
        # de adyacencia. Sera necesario? no el mismo KG nos esta otorgando dicha matriz?
        # al igual que otros metodos que quiza debamos de ignorar?
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
        elif method == "ageb":
            return none
        elif method == "alcaldia":
            return none
