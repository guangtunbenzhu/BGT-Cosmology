;+
; load and interpolate *ALL* quasar spectra
; output at both restframe and observer framer, with common central wavelength grid for each 
;-
pro jhusdss_hstfos_restobs_allqsospec, overwrite=overwrite, $
       doobserver=doobserver, dorest=dorest


;; qsopath
parent_path='~/SDATA/SDSS/AllInOne'
outfile_base = parent_path+'/AIO_QSO_HSTFOS'
wave_outfile = parent_path+'/AIO_CommonWave.fits'

channel = ['h130', 'h190', 'h270']
;; no header files, use explicit file names
 observer_outfile = outfile_base+'_ObserverFrame_Wave01100_03350A.fits'
rest_outfile_base = outfile_base+'_NEDzRestFrame.fits'

qsofile = '~/SDATA/Quasars/HSTFOS/table1_master.fits'
qso = mrdfits(qsofile, 1)
nqso = n_elements(qso)

if (file_test(wave_outfile)) then begin
   splog, "Warning: wavelength grid already exists. Use the existing one."
   wave = (mrdfits(wave_outfile, 1)).wave
   loglam = alog10(wave)
endif else begin
   loglam = jhusdss_get_loglam(minwave=448., maxwave=10404.)
   wave = 10.^loglam
   mwrfits, {wave:wave}, wave_outfile, /create
endelse
nwave = n_elements(loglam)

delta_z = 0.1
nzbin = fix((max(qso.z)+0.101)/delta_z)
print, "Memory used: ", memory(/current)/1024./1024., ' MB'

;; #######################
;; observer frame
if keyword_set(doobserver) then begin
;; #######################
print, "I am now working on the observer frame."

if (file_test(observer_outfile) and ~keyword_set(overwrite)) then begin
   splog, 'File exists. Use Overwrite to overwrite!'
   return
endif else begin
   splog, 'Will write into these files:'
   print, observer_outfile 
endelse

observer_wave_range = [1100., 3350.]
observer_loc = value_locate(wave, observer_wave_range)
observer_loglam = loglam[observer_loc[0]:observer_loc[1]+1]
observer_nwave = n_elements(observer_loglam)
observer_outstr = {ra:qso.ra2000, dec:qso.dec2000, z:qso.z, $
                   wave:wave[observer_loc[0]:observer_loc[1]+1], $
                   flux:fltarr(nqso, observer_nwave), $
                   ivar:fltarr(nqso, observer_nwave)}

observer_allflux_out = fltarr(nqso, observer_nwave)
observer_allivar_out = fltarr(nqso, observer_nwave)
print, "Memory used: ", memory(/current)/1024./1024., ' MB'

;; observer frame
for i=0L, nqso-1L do begin
    tmp_allflux = fltarr(3, n_elements(observer_loglam))
    tmp_allivar = fltarr(3, n_elements(observer_loglam))
    for j=0L, 2L do begin
        tmpspec = jhusdss_hstfos_readspec(qso[i], channel[j], status=status)
        if (status eq 1) then begin
           tmpwave = tmpspec.wave
           tmpflux = tmpspec.flux
           tmpivar = (1.-(tmpspec.error eq 0.))/(tmpspec.error^2 + (tmpspec.error eq 0.))
           tmpmask = (tmpivar le 0.)
           curr_loglam = alog10(tmpwave)
           combine1fiber, curr_loglam, tmpflux, tmpivar, newloglam=observer_loglam, $
                          newflux=finterp, newivar=iinterp, maxiter=0, $
                          finalmask=tmpmask, andmask=maskinterp
           tmp_allflux[j,*] = finterp
           tmp_allivar[j,*] = iinterp
        endif
    endfor
    ;; Co-add
    tmp_outivar = total(tmp_allivar, 1)
    tmp_outflux = total(tmp_allflux*tmp_allivar, 1)/(tmp_outivar+(tmp_outivar eq 0.))*(tmp_outivar gt 0.)
    observer_allflux_out[i,*] = tmp_outflux*10.
    observer_allivar_out[i,*] = tmp_outivar/100.
endfor

observer_outstr.flux = observer_allflux_out
observer_outstr.ivar = observer_allivar_out
mwrfits, observer_outstr, observer_outfile, /create
delvar, observer_allflux_out
delvar, observer_allvar_out
delvar, observer_outstr

;; #######################
endif ;; observer frame
;; #######################

;; #######################
;; rest frame
if keyword_set(dorest) then begin
;; #######################
print, "I am now working on the rest frame."

wave_range_aa = [450., 900.]
rest_loc_aa = value_locate(wave, wave_range_aa)
rest_outfile_aa = repstr(rest_outfile_base, '.fits', '_Wave00450_00900A.fits')
wave_range_bb = [900., 1800.]
rest_loc_bb = value_locate(wave, wave_range_bb)
rest_outfile_bb = repstr(rest_outfile_base, '.fits', '_Wave00900_01800A.fits')
wave_range_cc = [1800., 3600.]
rest_loc_cc = value_locate(wave, wave_range_cc)
rest_outfile_cc = repstr(rest_outfile_base, '.fits', '_Wave01800_03600A.fits')
wave_range_dd = [3600., 7200.]
rest_loc_dd = value_locate(wave, wave_range_dd)
rest_outfile_dd = repstr(rest_outfile_base, '.fits', '_Wave03600_07200A.fits')

if (file_test(rest_outfile_aa) and ~keyword_set(overwrite)) then begin
   splog, 'File exists. Use Overwrite to overwrite!'
   return
endif else begin
   splog, 'Will write into these files:'
   print, rest_outfile_aa
   print, rest_outfile_bb
   print, rest_outfile_cc
   print, rest_outfile_dd
endelse

;; two images, one for flux, one for ivar
rest_allflux_out = fltarr(nqso, nwave)
rest_allivar_out = fltarr(nqso, nwave)
print, "Memory used: ", memory(/current)/1024./1024., ' MB'

;; rest frame
;for i=0L, nqso-1L do begin
for i=0L, nqso-1L do begin
    rest_loc = ((value_locate(wave, [1100./(1.+qso[i].z), 3350./(1.+qso[i].z)]) > 0L) < (nwave-1L))
    rest_loglam = loglam[rest_loc[0]:rest_loc[1]+1]
    tmp_allflux = fltarr(3, n_elements(rest_loglam))
    tmp_allivar = fltarr(3, n_elements(rest_loglam))
    for j=0L, 2L do begin
        tmpspec = jhusdss_hstfos_readspec(qso[i], channel[j], status=status)
        if (status eq 1) then begin
           tmpwave = tmpspec.wave/(1.+qso[i].z)
           tmpflux = tmpspec.flux*(1.+qso[i].z)
           tmpivar = (1.-(tmpspec.error eq 0.))/(tmpspec.error^2 + (tmpspec.error eq 0.))
           tmpivar = tmpivar/(1.+qso[i].z)^2
           tmpmask = (tmpivar le 0.)
           curr_loglam = alog10(tmpwave)
           combine1fiber, curr_loglam, tmpflux, tmpivar, newloglam=rest_loglam, $
                          newflux=finterp, newivar=iinterp, maxiter=0, $
                          finalmask=tmpmask, andmask=maskinterp
           tmp_allflux[j,*] = finterp
           tmp_allivar[j,*] = iinterp
        endif
    endfor
    ;; Co-add
    tmp_outivar = total(tmp_allivar, 1)
    tmp_outflux = total(tmp_allflux*tmp_allivar, 1)/(tmp_outivar+(tmp_outivar eq 0.))*(tmp_outivar gt 0.)
    rest_allflux_out[i,rest_loc[0]:rest_loc[1]+1] = tmp_outflux*10.
    rest_allivar_out[i,rest_loc[0]:rest_loc[1]+1] = tmp_outivar/100.
endfor

;; rest frame
;; 450-900 AA
iaa =  where(qso.z gt (1100./wave_range_aa[1]-1.-0.001) and qso.z le (3350./wave_range_aa[0]-1.+0.001), naa)
if (naa gt 0) then begin
rest_outstr_aa = {index_qso:iaa, ra:qso[iaa].ra2000, dec:qso[iaa].dec2000, z: qso[iaa].z, $
                  wave:wave[rest_loc_aa[0]:rest_loc_aa[1]+1], $
                  flux:rest_allflux_out[iaa,rest_loc_aa[0]:rest_loc_aa[1]+1], $
                  ivar:rest_allivar_out[iaa,rest_loc_aa[0]:rest_loc_aa[1]+1]}
mwrfits, rest_outstr_aa, rest_outfile_aa, /create
delvar, rest_outstr_aa
endif

;; 900-1800 AA
ibb =  where(qso.z gt (1100./wave_range_bb[1]-1.-0.001) and qso.z le (3350./wave_range_bb[0]-1.+0.001), nbb)
if (nbb gt 0) then begin
rest_outstr_bb = {index_qso:ibb, ra:qso[ibb].ra2000, dec:qso[ibb].dec2000, z: qso[ibb].z, $
                  wave:wave[rest_loc_bb[0]:rest_loc_bb[1]+1], $
                  flux:rest_allflux_out[ibb,rest_loc_bb[0]:rest_loc_bb[1]+1], $
                  ivar:rest_allivar_out[ibb,rest_loc_bb[0]:rest_loc_bb[1]+1]}
mwrfits, rest_outstr_bb, rest_outfile_bb, /create
delvar, rest_outstr_bb
endif

;; 1800 - 3600 AA
icc =  where(qso.z gt (1100./wave_range_cc[1]-1.-0.001) and qso.z le (3350./wave_range_cc[0]-1.+0.001), ncc)
if (ncc gt 0) then begin
rest_outstr_cc = {index_qso:icc, ra:qso[icc].ra2000, dec:qso[icc].dec2000, z: qso[icc].z, $
                  wave:wave[rest_loc_cc[0]:rest_loc_cc[1]+1], $
                  flux:rest_allflux_out[icc,rest_loc_cc[0]:rest_loc_cc[1]+1], $
                  ivar:rest_allivar_out[icc,rest_loc_cc[0]:rest_loc_cc[1]+1]}
mwrfits, rest_outstr_cc, rest_outfile_cc, /create
delvar, rest_outstr_cc
endif

;; 3600 - 7200 AA
idd =  where(qso.z gt (1100./wave_range_dd[1]-1.-0.001) and qso.z le (3350./wave_range_dd[0]-1.+0.001), ndd)
if (ndd gt 0) then begin
rest_outstr_dd = {index_qso:idd, ra:qso[idd].ra2000, dec:qso[idd].dec2000, z: qso[idd].z, $
                  wave:wave[rest_loc_dd[0]:rest_loc_dd[1]+1], $
                  flux:rest_allflux_out[idd,rest_loc_dd[0]:rest_loc_dd[1]+1], $
                  ivar:rest_allivar_out[idd,rest_loc_dd[0]:rest_loc_dd[1]+1]}
mwrfits, rest_outstr_dd, rest_outfile_dd, /create
delvar, rest_outstr_dd
endif

;; #######################
endif ;; rest frame
;; #######################

end
