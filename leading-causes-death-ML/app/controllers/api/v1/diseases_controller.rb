class Api::V1::DiseasesController < ApplicationController
    def index
        @diseases = Disease.select("id, year, leading_cause, sum(deaths) AS deaths, sex").group("year, leading_cause, sex")
        render json: @diseases
    end
    def show
        @disease = Disease.find(params[:id])
        render json: @disease
    end
    def create
        @disease = Disease.create(year: params[:year], leading_cause: params[:leading_cause], sex: params[:sex], deaths: params[:deaths])
        render json: @disease
    end
    def update
        @disease = Disease.find(params[:id])
        @disease.update(year: params[:year], leading_cause: params[:leading_cause], sex: params[:sex], deaths: params[:deaths])
        render json: @disease
    end
end
