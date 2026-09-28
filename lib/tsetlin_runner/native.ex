TsetlinRunner.Target.configure_cargo_target!()

defmodule TsetlinRunner.Native do
  @moduledoc false
  use Rustler, otp_app: :tsetlin_runner, crate: "tsetlin_nif"

  def load_model_nif(_path), do: :erlang.nif_error(:nif_not_loaded)
  def predict_nif(_resource, _packed_bits), do: :erlang.nif_error(:nif_not_loaded)

  def classify_frame_nif(_resource, _jpeg, _out_w, _out_h, _radius),
    do: :erlang.nif_error(:nif_not_loaded)
end
