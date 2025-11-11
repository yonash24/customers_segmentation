from DataHandler import ImportData, DataAnalysis



def main():
    ### import the data and transfer it to data frame
    ImportData.import_data()
    df = ImportData.to_df()

    ### analys the data and understand it ###
    analizer = DataAnalysis(df)
    analizer.get_edges_info()
    analizer.get_miss_vals_amount()
    analizer.get_numeric_statistical_info()
    analizer.get_categoric_unique_info()
    analizer.features_hist()
    analizer.top_ten_countries_graph()
    analizer.featrues_correlation()
    analizer.distribution_detection()
    analizer.outlier_detection()

    




if __name__ == "__main__":
    main()