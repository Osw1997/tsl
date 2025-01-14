
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

    def __init__(self, root=None, impute_zeros=True, freq=None, geo_detail="alcaldia", crime_dataset=None):
        # set root path
        self.root = root
        self.geo_detail = geo_detail
        
        # Check if desired crime to process is in dataset
        available_crimes = ['<http://localhost/FRAUDE>',
           '<http://localhost/VIOLENCIA_FAMILIAR>',
           '<http://localhost/ROBO_DE_OBJETOS>',
           '<http://localhost/ROBO_DE_OBJETOS_DEL_INTERIOR_DE_UN_VEHICULO>',
           '<http://localhost/ROBO_A_TRANSEUNTE_EN_VIA_PUBLICA_CON_VIOLENCIA>',
           '<http://localhost/ROBO_DE_ACCESORIOS_DE_AUTO>',
           '<http://localhost/ABUSO_DE_CONFIANZA>',
           '<http://localhost/ROBO_A_NEGOCIO_SIN_VIOLENCIA_POR_FARDEROS_(TIENDAS_DE_AUTOSERVICIO)>',
           '<http://localhost/AMENAZAS>',
           '<http://localhost/USURPACIÓN_DE_IDENTIDAD>'
        ]
        self.crime_dataset = crime_dataset
        self.crime = crime_dataset.split("/")[-1].split(">")[0]
        if self.crime_dataset not in (available_crimes):
            raise ValueError('Desired crime not in list: "%s"' % (self.crime_dataset))
    
        print(f">>> Crime to process: {self.crime_dataset}")
        
        # load dataset
        # df, dist, mask, df_raw = self.load(impute_zeros=impute_zeros)
        df, dist, mask = self.load(impute_zeros=impute_zeros)
        # return df
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

    # En esencia los argumentos de esta function/metodo no deberian ded existir
    # puesto que ambos elementos deberán ser atributos del objeto previamente
    # declarados.
    def contained(self, geo_df, points_series):
        # Se obtienen los indices/nombre de las alcaldias a la que pertenecen cada punto
        # el conteo puede llevarse a cabo despues con un built-in method "group-by"
        list_belonging = list(map(geo_df['geometry'].contains, points_series))
        index_geom = [-1 if x.sum() == 0 else geo_df[x].index[0] for x in list_belonging]
        name_geo = ['NA' if x.sum() == 0 else geo_df[x]['nomgeo'].item() for x in list_belonging]
        return index_geom, name_geo

    #### La matriz de sitancias entre denuncias y denuncias denuncias las podemos calcular asi
    def distance_matrix_crimes(self, df):
        #cambiamos la proyección para distancias en metros
        df = gpd.GeoDataFrame(df).to_crs('EPSG:3857')
#         df = df.set_crs('EPSG:4326').to_crs('EPSG:3857')
        dist = df.geometry.apply(lambda g: df.centroid.distance(g.centroid))

        # FOR TESTING PURPOSES
        dist.to_csv("/content/tsl/tsl/datasets/raw_files_to_remove/dist_csv.csv")

        # # Lets follow the same pattern that is in the other datasets
        # dist[dist == 0] = np.inf
        
        # Save to built directory (Update in the class definition)
        # path = os.path.join(self.root_dir, 'metr_la_dist.npy')
        path = "crime_cdmx_dist.npy"
        np.save(path, dist)

    def load_ttl_sparql(self):
        NAMESPACES = {
            "wtm":  "http://purl.org/heals/food/",
            "ind":  "http://purl.org/heals/ingredient/",
            "recipe":  "https://www.food.com/recipe/",
    
            "crime": "http://localhost/ontology2#",
            "cube": "http://purl.org/linked-data/cube#",
            "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
            "ontology": "http://dbpedia.org/ontology/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#"
        }
    
        kg = kglab.KnowledgeGraph(namespaces = NAMESPACES)
        print(f"--> CURRENT WORK DIRECTORY: [{os.getcwd()}]")
        # _ = kg.load_rdf("/content/tsl/tsl/datasets/raw_files_to_remove/testind3.ttl")
        _ = kg.load_rdf("/content/tsl/tsl/datasets/raw_files_to_remove/testindtop10.ttl")
        
        print("CURRENT NAMESPASE: ")
        kg.describe_ns()
    
        # Buscamos tener id, delito, atributos, coordenadas y fecha en el DF
        # A clasificar: crime:tieneCategoria
        # Atributo: crime:contiene, crime:edad, crime:genero
        # sparql_contiene = """
        #     SELECT distinct *
        #     WHERE {
        #         ?uri a crime:obs .
        #         ?uri a ?type .   
        #         ?uri crime:tieneFecha ?date .
        #         ?uri geo1:lat ?lat .
        #         ?uri geo1:long ?long .
    
        #         ?uri crime:contiene ?atribute .
        #     }
        # """
        sparql_categoria = """
            SELECT distinct *
            WHERE {
                ?uri a crime:obs .
                ?uri a ?type .   
                ?uri crime:tieneFecha ?date .
                ?uri geo1:lat ?lat .
                ?uri geo1:long ?long .
    
                ?uri crime:tieneCategoria ?atribute .
            }
        """
        sparql_edad = """
            SELECT distinct *
            WHERE {
                ?uri a crime:obs .
                ?uri a ?type .   
                ?uri crime:tieneFecha ?date .
                ?uri geo1:lat ?lat .
                ?uri geo1:long ?long .
    
                ?uri crime:edad ?atribute .
            }
        """
        sparql_genero = """
            SELECT distinct *
            WHERE {
                ?uri a crime:obs .
                ?uri a ?type .   
                ?uri crime:tieneFecha ?date .
                ?uri geo1:lat ?lat .
                ?uri geo1:long ?long .
    
                ?uri crime:genero ?atribute
            }
        """
    
        # df_contiene = kg.query_as_df(sparql=sparql_contiene)
        df_categoria = kg.query_as_df(sparql=sparql_categoria)
        df_edad = kg.query_as_df(sparql=sparql_edad)
        df_genero = kg.query_as_df(sparql=sparql_genero)
    
        # df_atributes = pd.concat([df_contiene, df_edad, df_genero], ignore_index=True)
        df_atributes = pd.concat([df_categoria, df_edad, df_genero], ignore_index=True)
        df_atributes = df_atributes.groupby(['uri', 'date', 'type', 'long', 'lat'])['atribute'].apply(list).reset_index(name='atribute')
        # df_atributes
        
        # Luego obtengamos el uri, delito a clasificar
        # sparql_crimes = """
        #     SELECT distinct *
        #     WHERE {
        #         ?uri crime:tieneCategoria ?crime .
        #     }
        # """
        sparql_crimes = """
            SELECT distinct *
            WHERE {
                ?uri crime:contiene """ + self.crime_dataset + """ .
                ?uri crime:contiene ?crime .
            }
        """
        df_crimes = kg.query_as_df(sparql=sparql_crimes)
        # Join by uri
        df = pd.merge(df_atributes, df_crimes, how="inner", on="uri")
        # Se le crea la geometria "Point" a cada crimen segun su longitud/latitud
        df["geometry"] = df[["long", "lat"]].T.apply(Point)
        
        return df 

    def load_raw(self):
        path_ttl_sparql = f"/content/drive/MyDrive/tsl_datasets/processed_ttl_sparql_{self.crime}.csv"
        if os.path.isfile(path_ttl_sparql):
            df = pd.read_csv(path_ttl_sparql)
            # Convert "geometry" column <str> to <geopandas>
            df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
        else:
            df = self.load_ttl_sparql()
            df.to_csv(path_ttl_sparql, index=False)
            
        # Carga geometrias de alcaldias
        if self.geo_detail == 'alcaldia':
            path_geojson = "/content/tsl/tsl/datasets/raw_files_to_remove/alcaldias_cdmx.json"
        elif self.geo_detail == 'ageb':
            path_geojson = "/content/tsl/tsl/datasets/raw_files_to_remove/agebs.json"
        f = open(path_geojson, encoding='utf8')
        
        
        json_alcaldias = json.load(f)
        # geo_df = gpd.GeoDataFrame.from_features(json_alcaldias["features"])
        geo_df = gpd.GeoDataFrame.from_features(json_alcaldias["features"], crs='EPSG:4326')

        # TO REMOVE ONCE JSON HAS TE APPROPRIATE FORMAT (agebs)
        # ========================================================================================
        # ========================================================================================
        if self.geo_detail == 'ageb':
            geo_df["nomgeo"] = geo_df["id"] + "_" + geo_df["cve_ageb"]
            del geo_df["id"], geo_df["cve_ageb"]
        # ========================================================================================
        # ========================================================================================
        geo_df = geo_df.sort_values("nomgeo")
        
        # # Encuentra su respectiva alcaldia y las respectivas coordenadas a evaluar
        # if aggregation_level == "alcaldia":
        # Se invoca la funcion y se le agrega la columnas al dataframe
        index_geom, name_geo = self.contained(geo_df, df["geometry"])
        df["index_alcaldia"] = index_geom
        df["nombre_alcaldia"] = name_geo
        # Let's remove empty lat,long/Invalid rows
        valid_rows = (df["index_alcaldia"] != -1)
        df = df[valid_rows]
        
        # Lets convert the dataframe into another dataframe but using right format
        df_raw = df.copy() # TO REMOVE: Only for testing purposes
        df = df.value_counts(["date", "nombre_alcaldia"]).unstack(fill_value=0)
        df = df.set_index(pd.DatetimeIndex(df.index))
        df = df.resample('D').sum()
        
        # Agregacion semanal
        df['date'] = df.index - pd.to_timedelta(7, unit='d')
        df = df.groupby([pd.Grouper(key='date', freq='W')]).sum()
        # # Agregacion mensual
        # df['date'] = df.index - pd.DateOffset(months=1)
        # df = df.groupby([pd.Grouper(key='date', freq='M')]).sum()

        # FOR TESTING PURPOSES (GPU/CPU LIMIT)
        sorted_columns = df.sum().sort_values(ascending=False, inplace=False)
        # df.loc[:,s.index]
        df = df[sorted_columns.index[:1000]]

        # Con esta funcion obtenemos la distancia de cada denuncia con respecto a cada denuncia
        # Debido a que no todos las geometrias pueden ser usadas, existira una
        # incogruencia entre la matriz de distancias respecto a las columnas
        # del dataframe ocasionando un error al agregar la matriz de covarianza
        # por lo que se filtran el dataframe de geometrias con las geometrias que 
        # se utilizaron.
        used_geometries = list(df.columns.get_level_values(0))
        print(f"USed geometries: {used_geometries}")

        geo_df.index = geo_df["nomgeo"]
        geo_df = geo_df[geo_df["nomgeo"].isin(used_geometries)]
        geo_df = geo_df.reindex(used_geometries)
        
        # Con esta funcion obtenemos la distancia de cada denuncia con respecto a cada denuncia
        # una vez filtradas las geometrias necesarias.
        self.distance_matrix_crimes(geo_df)
        # Carga la matriz de distancias desde el archivo .npy
        # path = os.path.join(self.root_dir, 'crime_cdmx_dist.npy')
        path = 'crime_cdmx_dist.npy'
        dist = np.load(path)

        # Filter rows < 2017
        df = df[df.index >= '2017-01-01']

        return df, dist
        # return df, dist, df_raw

    def load(self, impute_zeros=True):
        # Aqui se va a tener que hacer un poco de pre-procesamiento 
        # para la tabla creada a partir de la consulta SPARQL.
        processed_ttl_path = f"/content/drive/MyDrive/tsl_datasets/processed_ttl_{self.crime}.csv"
        processed_ttl_distance_path = f"/content/drive/MyDrive/tsl_datasets/processed_ttl_distance_{self.crime}.npy"
        if os.path.isfile(processed_ttl_path) and os.path.isfile(processed_ttl_distance_path):
            print("Loading processed data from npy file")
            # Load multiindex dataframe 
            df = pd.read_csv(processed_ttl_path)
            # df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
            # Load distance matrix
            dist = np.load(processed_ttl_distance_path)
        else:
            print("Processing TTL and matrix distance")
            df, dist = self.load_raw() # TO REMOVE: Only for testing purposes
            df.to_csv(processed_ttl_path, index=True)            
            np.save(processed_ttl_distance_path, dist)
            
        # df, dist = self.load_raw()
        mask = (df.values != 0.).astype('uint8')
        if impute_zeros:
            df = df.replace(to_replace=0., method='ffill')
        
        # return df, dist, mask, df_raw # TO REMOVE: Only for testing purposes
        return df, dist, mask

    # TODO:
    # UNA VEZ QUE PANCHO ME HAYA PASADO LOS DATAFRAMES
    # - CARGARLO A ESTE CODIGO [OK]
    # - ARMONIZAR EL FORMATO DE LAS TABLAS [OK]
    # - CALCULAR LA SIMILITUD DE LOS NODOS MEDIANTE DIFERENTES METODOS (GAUSSIAN KERNEL) [OK]
    
    # - ES CORRECTA LA IMPLEMENTACION PARA SIMILITUD A NIVEL DE AGEB Y ALCALDIA? COMO RELACIONARLOS?
    # - IMPLEMENTAR METODO DE SIMILITUD PARA NIVEL alcaldia [TODO]
    # distancia_ageb, d_alcaldaia

    # probar con el ttl 
    # ejemplo con red neurnal

    # - Crear VM con todas las bibliotecas necesarias[OK]

    
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
