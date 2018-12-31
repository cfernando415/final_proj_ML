class DiseaseSerializer < ActiveModel::Serializer
  attributes :id, :year, :leading_cause, :deaths, :sex
end