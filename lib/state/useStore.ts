import { create } from 'zustand'

/** Datos del formulario introducidos por el usuario */
export interface FormData {
  municipio: string
  intereses: string[]
  numeric: Record<string, number>
  years: number
  growthRate: number
}

/** Resultado de la inferencia */
export interface ResultData {
  predictions: number[]
  yearsList: number[]
  recommended: { cultivo: string; mean_production: number }[]
  cluster: number | null
  cultivo?: string
}

interface StoreState {
  formData: FormData | null
  resultData: ResultData | null
  setFormData: (data: FormData) => void
  setResultData: (data: ResultData) => void
}

export const useStore = create<StoreState>((set) => ({
  formData: null,
  resultData: null,
  setFormData: (data) => set({ formData: data }),
  setResultData: (data) => set({ resultData: data }),
}))