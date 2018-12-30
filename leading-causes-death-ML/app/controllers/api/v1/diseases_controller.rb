class Api::V1::DiseasesController < ApplicationController
    def index
        @diseases = Disease.select("year, leading_cause, sum(deaths) AS deaths, sex").group("year, leading_cause, sex")
        render json: @diseases
    end
end
