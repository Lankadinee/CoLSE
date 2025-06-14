create index idx_global_active_power on power using btree(Global_active_power);
create index idx_global_reactive_power on power using btree(Global_reactive_power);
create index idx_voltage on power using btree(Voltage);
create index idx_global_intensity on power using btree(Global_intensity);
create index idx_sub_metering_1 on power using btree(Sub_metering_1);
create index idx_sub_metering_2 on power using btree(Sub_metering_2);
create index idx_sub_metering_3 on power using btree(Sub_metering_3);