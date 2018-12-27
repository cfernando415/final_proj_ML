class DiseaseSerializer < ActiveModel::Serializer
  attributes :year, :leading_cause, :deaths, :sex
end