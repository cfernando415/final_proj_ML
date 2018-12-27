class Api::V1::DiseasesController < ApplicationController
    def index
        @diseases = Disease.all
        render json: @diseases
    end
end
